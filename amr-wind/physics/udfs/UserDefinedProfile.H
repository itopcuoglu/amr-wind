#ifndef USERDEFINEDPROFILE_H
#define USERDEFINEDPROFILE_H

#include "AMReX_Geometry.H"
#include "AMReX_Gpu.H"

namespace amr_wind {

class Field;

namespace udf {

struct UserDefinedProfile
{
    struct DeviceOp
    {
        int idir{2};
        amrex::Real hmin;
        amrex::Real hmax;
        int npts;
        amrex::Real deltah;
        amrex::Gpu::DeviceVector<amrex::Real> prof_h, prof_vec;

        AMREX_GPU_DEVICE
        inline void operator()(
            const amrex::IntVect& iv,
            amrex::Array4<amrex::Real> const& field,
            amrex::GeometryData const& geom,
            const amrex::Real /*unused*/,
            amrex::Orientation /*unused*/,
            const int comp,
            const int dcomp,
            const int orig_comp) const
        {
            const auto* problo = geom.ProbLo();
            const auto* dx = geom.CellSize();
            const auto zco = problo[idir] + (iv[idir] + 0.5) * dx[idir];

            const amrex::Real* h_ptr = prof_h.data();
            const amrex::Real* v_ptr = prof_vec.data();
            amrex::Real val;
            int npt_l = std::floor(zco / deltah);
            // amrex::Real h_l = prof_h[npt_l];
            amrex::Real h_l = h_ptr[npt_l];

            if (npt_l > npts - 2) {
                // val = prof_vec[npts - 1 + npts * comp];
                val = v_ptr[npts - 1 + npts * comp];
            } else if (npt_l < 0) {
                // val = prof_vec[0 + npts * comp];
                val = v_ptr[0 + npts * comp];
            } else {
                // val = prof_vec[npt_l + npts * comp] +
                //       (prof_vec[npt_l + 1 + npts * comp] -
                //        prof_vec[npt_l + npts * comp]) *
                //           (zco - h_l) / deltah;
                val = v_ptr[npt_l + npts * comp] +
                      (v_ptr[npt_l + 1 + npts * comp] -
                       v_ptr[npt_l + npts * comp]) *
                          (zco - h_l) / deltah;
            }

            field(iv[0], iv[1], iv[2], dcomp + comp) = val;
        }
    };

    using DeviceType = DeviceOp;

    static std::string identifier() { return "UserDefinedProfile"; }

    explicit UserDefinedProfile(const Field& fld);

    DeviceType device_instance() const { return m_op; }

    DeviceOp m_op;

}; // namespace udf

} // namespace udf
} // namespace amr_wind

#endif /* USERDEFINEDPROFILE_H */
