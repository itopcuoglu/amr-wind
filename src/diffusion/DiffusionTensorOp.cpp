#include <incflo.H>
#include <DiffusionTensorOp.H>
#include <AMReX_ParmParse.H>
#include "diffusion.H"

using namespace amrex;

DiffusionTensorOp::DiffusionTensorOp(incflo* a_incflo) : m_incflo(a_incflo)
{
    readParameters();

    int finest_level = m_incflo->finestLevel();

    LPInfo info_solve;
    info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);
    LPInfo info_apply;
    info_apply.setMaxCoarseningLevel(0);

    m_reg_solve_op.reset(new MLTensorOp(
        m_incflo->Geom(0, finest_level), m_incflo->boxArray(0, finest_level),
        m_incflo->DistributionMap(0, finest_level), info_solve));
    m_reg_solve_op->setMaxOrder(m_mg_maxorder);
    m_reg_solve_op->setDomainBC(
        diffusion::get_diffuse_tensor_bc(m_incflo->velocity(), Orientation::low),
        diffusion::get_diffuse_tensor_bc(m_incflo->velocity(), Orientation::high));
    if (m_incflo->need_divtau()) {
        m_reg_apply_op.reset(new MLTensorOp(
            m_incflo->Geom(0, finest_level),
            m_incflo->boxArray(0, finest_level),
            m_incflo->DistributionMap(0, finest_level), info_apply));
        m_reg_apply_op->setMaxOrder(m_mg_maxorder);
        m_reg_apply_op->setDomainBC(
            diffusion::get_diffuse_tensor_bc(m_incflo->velocity(), Orientation::low),
            diffusion::get_diffuse_tensor_bc(m_incflo->velocity(), Orientation::high));
    }
}

void DiffusionTensorOp::readParameters()
{
    ParmParse pp("diffusion");

    pp.query("verbose", m_verbose);
    pp.query("mg_verbose", m_mg_verbose);
    pp.query("mg_cg_verbose", m_mg_cg_verbose);
    pp.query("mg_max_iter", m_mg_max_iter);
    pp.query("mg_cg_maxiter", m_mg_cg_maxiter);
    pp.query("mg_max_fmg_iter", m_mg_max_fmg_iter);
    pp.query("mg_max_coarsening_level", m_mg_max_coarsening_level);
    pp.query("mg_maxorder", m_mg_maxorder);
    pp.query("mg_rtol", m_mg_rtol);
    pp.query("mg_atol", m_mg_atol);
    pp.query("bottom_solver_type", m_bottom_solver_type);
}

void DiffusionTensorOp::diffuse_velocity(
    Vector<MultiFab*> const& velocity,
    Vector<MultiFab const*> const& density,
    Vector<MultiFab const*> const& eta,
    Real dt)
{
    //
    //      alpha a - beta div ( b grad )   <--->   rho - dt div ( mu grad )
    //
    // So the constants and variable coefficients are:
    //
    //      alpha: 1
    //      beta: dt
    //      a: rho
    //      b: mu

    if (m_verbose > 0) {
        amrex::Print() << "Diffusing velocity components all together..."
                       << std::endl;
    }

    const int finest_level = m_incflo->finestLevel();

    m_reg_solve_op->setScalars(1.0, dt);
    for (int lev = 0; lev <= finest_level; ++lev) {
        m_reg_solve_op->setACoeffs(lev, *density[lev]);
        Array<MultiFab, AMREX_SPACEDIM> b =
            diffusion::average_velocity_eta_to_faces(m_incflo->Geom(lev), *eta[lev]);
        m_reg_solve_op->setShearViscosity(lev, GetArrOfConstPtrs(b));
        m_reg_solve_op->setLevelBC(lev, velocity[lev]);
    }

    Vector<MultiFab> rhs(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        rhs[lev].define(
            velocity[lev]->boxArray(), velocity[lev]->DistributionMap(),
            AMREX_SPACEDIM, 0);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(rhs[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& rhs_a = rhs[lev].array(mfi);
            Array4<Real const> const& vel_a = velocity[lev]->const_array(mfi);
            Array4<Real const> const& rho_a = density[lev]->const_array(mfi);
            amrex::ParallelFor(
                bx, AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                    rhs_a(i, j, k, n) = rho_a(i, j, k) * vel_a(i, j, k, n);
                });
        }
    }

    MLMG mlmg(*m_reg_solve_op);

    // The default bottom solver is BiCG
    if (m_bottom_solver_type == "smoother") {
        mlmg.setBottomSolver(MLMG::BottomSolver::smoother);
    } else if (m_bottom_solver_type == "hypre") {
        mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
    }
    // Maximum iterations for MultiGrid / ConjugateGradients
    mlmg.setMaxIter(m_mg_max_iter);
    mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
    mlmg.setCGMaxIter(m_mg_cg_maxiter);

    // Verbosity for MultiGrid / ConjugateGradients
    mlmg.setVerbose(m_mg_verbose);
    mlmg.setCGVerbose(m_mg_cg_verbose);

    mlmg.solve(velocity, GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
}

void DiffusionTensorOp::compute_divtau(
    Vector<MultiFab*> const& a_divtau,
    Vector<MultiFab*> const& velocity,
    Vector<MultiFab const*> const& a_density,
    Vector<MultiFab const*> const& a_eta)
{
    BL_PROFILE("DiffusionTensorOp::compute_divtau");

    int finest_level = m_incflo->finestLevel();

    // We want to return div (mu grad)) phi
    m_reg_apply_op->setScalars(0.0, -1.0);
    for (int lev = 0; lev <= finest_level; ++lev) {
        m_reg_apply_op->setACoeffs(lev, *a_density[lev]);
        Array<MultiFab, AMREX_SPACEDIM> b =
            diffusion::average_velocity_eta_to_faces(m_incflo->Geom(lev), *a_eta[lev]);
        m_reg_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(b));
        m_reg_apply_op->setLevelBC(lev, velocity[lev]);
    }

    MLMG mlmg(*m_reg_apply_op);
    mlmg.apply(a_divtau, velocity);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (int lev = 0; lev <= finest_level; ++lev) {
        for (MFIter mfi(*a_divtau[lev], TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& divtau_arr = a_divtau[lev]->array(mfi);
            Array4<Real const> const& rho_arr =
                a_density[lev]->const_array(mfi);
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    Real rhoinv = 1.0 / rho_arr(i, j, k);
                    divtau_arr(i, j, k, 0) *= rhoinv;
                    divtau_arr(i, j, k, 1) *= rhoinv;
                    divtau_arr(i, j, k, 2) *= rhoinv;
                });
        }
    }
}
