#include "amr-wind/physics/udfs/UserDefinedProfile.H"
#include "amr-wind/core/Field.H"
#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/core/vs/vector.H"
#include "amr-wind/equation_systems/icns/icns.H"

#include "AMReX_ParmParse.H"

namespace amr_wind::udf {

UserDefinedProfile::UserDefinedProfile(const Field& fld)
{
    AMREX_ALWAYS_ASSERT(fld.name() == pde::ICNS::var_name());
    AMREX_ALWAYS_ASSERT(fld.num_comp() == AMREX_SPACEDIM);

    std::string prfl_file;
    amrex::ParmParse pp("UserDefinedProfile");
    pp.query("file_input", prfl_file);
    pp.query("direction", m_op.idir);
    pp.query("hmin", m_op.hmin);
    pp.query("hmax", m_op.hmax);
    pp.query("deltah", m_op.deltah);

    std::ifstream pp_infile;
    int npts;
    pp_infile.open(prfl_file.c_str(), std::ios_base::in);
    pp_infile >> npts;
    amrex::Vector<amrex::Real> prof_h, prof_u, prof_v, prof_w, prof_vec;

    prof_h.resize(npts);
    prof_u.resize(npts);
    prof_v.resize(npts);
    prof_w.resize(npts);
    m_op.prof_h.resize(npts);
    m_op.prof_vec.resize(3 * npts);
    m_op.prof_h_d.resize(npts);
    m_op.prof_vec_d.resize(3 * npts);

    m_op.npts = npts;

    int i;
    for (i = 0; i < npts; i++) {
        pp_infile >> prof_h[i] >> prof_u[i] >> prof_v[i] >> prof_w[i];
    }
    pp_infile.close();

    for (i = 0; i < npts; i++) {
        m_op.prof_h[i] = prof_h[i];
    }

    for (i = 0; i < npts; i++) {
        m_op.prof_vec[i] = prof_u[i];
    }

    for (i = 0; i < npts; i++) {
        m_op.prof_vec[i + 1 * npts] = prof_v[i];
    }

    for (i = 0; i < npts; i++) {
        m_op.prof_vec[i + 2 * npts] = prof_w[i];
    }

    AMREX_ALWAYS_ASSERT(m_op.prof_vec.size() == AMREX_SPACEDIM * npts);

    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, m_op.prof_h.begin(), m_op.prof_h.end(),
        m_op.prof_h_d.begin());

    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, m_op.prof_vec.begin(), m_op.prof_vec.end(),
        m_op.prof_vec_d.begin());

    m_op.h_ptr = m_op.prof_h_d.data();
    m_op.v_ptr = m_op.prof_vec_d.data();
}

} // namespace amr_wind::udf
