#ifndef VECTOR_SYSTEM_HPP_ // NOLINT
#define VECTOR_SYSTEM_HPP_
// ABOUTME: This file defines a VectorSystem class for advecting generic vector fields.
// ABOUTME: Specialized for spatially constant velocity and wavespeeds to simplify computations.
//==============================================================================
// AMRAdvection
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file vector_system.hpp
/// \brief Defines a class for solving vector field advection equations.
///

// c++ headers

// library headers

// internal headers
#include "AMReX_BLProfiler.H"
#include "AMReX_GpuControl.H"
#include "AMReX_ParmParse.H"

#include "hyperbolic_system.hpp"

AMREX_ENUM(VectorAvgType, Simple, Upwind); // NOLINT

/// Class for a vector field advection system with constant velocity
template <typename problem_t> class VectorSystem : public HyperbolicSystem<problem_t>
{
      using arrayconst_t = amrex::Array4<const amrex::Real> const;
      using array_t = amrex::Array4<amrex::Real>;
  
      public:
        constexpr static int vector_index = 0;
        static constexpr int nvar_per_dim_ = 1; // number of components per dimension for vector field
  
        static void ComputeFlux(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_flux_components, amrex::MultiFab const &cc_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, 
			       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &advection_velocity,
			       int reconstructionOrder, VectorAvgType vector_avg_type);

	static void ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_cValid, int reconstructionOrder);

	static void SolveAdvectionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_flux_mf, double dt, 
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);
};

template <typename problem_t>
void VectorSystem<problem_t>::ComputeFlux(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_flux_components, amrex::MultiFab const &cc_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &advection_velocity,
				      int reconstructionOrder, VectorAvgType vector_avg_type)
{
	const BL_PROFILE("VectorSystem::ComputeFlux()");
	const int nghost_cc = 4; // we only need 4 cc ghost cells when reconstructing cc->fc->ec using PPM

	// loop over each box-array on this level
	for (amrex::MFIter mfi(cc_mf_cVars); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// For constant velocity, we can directly compute the flux at edges
		// without the complex velocity field reconstruction needed for MHD
		
		// indexing: field[3: x-component/x-face]
		// create a view of all the vector field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_Vx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, VectorSystem<problem_t>::vector_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, VectorSystem<problem_t>::vector_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, VectorSystem<problem_t>::vector_index, 1),
		};

		// compute the flux through each cell-edge
		for (int iedge = 0; iedge < 3; ++iedge) {
			// define the two directions for the edge
			std::array<int, 2> extrap_dirs = {(iedge + 1) % 3, (iedge + 2) % 3};
			std::array<amrex::IntVect, 2> vecs_cc2ec = {amrex::IntVect::TheDimensionVector(extrap_dirs[0]),
								    amrex::IntVect::TheDimensionVector(extrap_dirs[1])};
			const amrex::IntVect vec_cc2ec = vecs_cc2ec[0] + vecs_cc2ec[1];
			const amrex::Box box_ec = amrex::convert(box_cc, vec_cc2ec);

			// indexing: field[2: i-compnent][2: i-side of edge]
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Vi_ieside;

			// define quantities
			for (int icomp = 0; icomp < 2; ++icomp) {
				ec_fabs_Vi_ieside[icomp][0].resize(box_ec, 1);
				ec_fabs_Vi_ieside[icomp][1].resize(box_ec, 1);
			}

			// extrapolate the two required face-centered vector field components to the cell-edge
			for (int icomp = 0; icomp < 2; ++icomp) {
				const int extrap_dir2edge = extrap_dirs[(icomp + 1) % 2];
				const auto dir2edge = static_cast<FluxDir>(extrap_dir2edge);
				const int wcomp = extrap_dirs[icomp];
				const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(wcomp);
				const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
				// extrapolate face-centered vector components to the cell-edge
				auto fc_array = fc_fabs_Vx[wcomp].const_array();
				auto lstate_array = ec_fabs_Vi_ieside[icomp][0].array();
				auto rstate_array = ec_fabs_Vi_ieside[icomp][1].array();
				VectorSystem<problem_t>::ReconstructTo(dir2edge, fc_array, lstate_array,
								    rstate_array, box_fc, reconstructionOrder);
			}

			// extract both components of vector field either side of the cell-edge
			const auto &V0_m = ec_fabs_Vi_ieside[0][0].const_array();
			const auto &V0_p = ec_fabs_Vi_ieside[0][1].const_array();
			const auto &V1_m = ec_fabs_Vi_ieside[1][0].const_array();
			const auto &V1_p = ec_fabs_Vi_ieside[1][1].const_array();

			// compute flux on the cell-edge for constant velocity advection
			const auto &flux_edge = ec_mf_flux_components[iedge][mfi].array();
			
			// Get constant velocity components for this edge direction
			const amrex::Real u0 = advection_velocity[extrap_dirs[0]];
			const amrex::Real u1 = advection_velocity[extrap_dirs[1]];

			if (vector_avg_type == VectorAvgType::Simple) {
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					// Simple averaging for constant velocity case
					const double V0_avg = 0.5 * (V0_m(i, j, k) + V0_p(i, j, k));
					const double V1_avg = 0.5 * (V1_m(i, j, k) + V1_p(i, j, k));
					// Flux is u × V for the component perpendicular to the edge
					flux_edge(i, j, k) = u0 * V1_avg - u1 * V0_avg;
				});
			} else if (vector_avg_type == VectorAvgType::Upwind) {
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					// Upwind flux for constant velocity case
					const double V0_flux = (u0 > 0.0) ? V0_m(i, j, k) : V0_p(i, j, k);
					const double V1_flux = (u1 > 0.0) ? V1_m(i, j, k) : V1_p(i, j, k);
					// Flux is u × V for the component perpendicular to the edge
					flux_edge(i, j, k) = u0 * V1_flux - u1 * V0_flux;
				});
			}
		}
	}
}

template <typename problem_t>
void VectorSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_cValid,
					 int reconstructionOrder)
{
	const BL_PROFILE("VectorSystem::ReconstructTo()");
	amrex::Box const &box_r = amrex::grow(box_cValid, 1);
	amrex::Box const &box_r_x1 = amrex::surroundingNodes(box_r, static_cast<int>(dir));
	if (reconstructionOrder == 4) {
		// note: only box_r is used. box_r_x1 is unused.
		switch (dir) {
			case FluxDir::X1:
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X1>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X2:
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X2>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X3:
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X3>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
		}
	} else if (reconstructionOrder == 3) {
		switch (dir) {
			case FluxDir::X1:
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X1>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X2:
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X2>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X3:
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X3>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
		}
	} else if (reconstructionOrder == 1) {
		switch (dir) {
			case FluxDir::X1:
				HyperbolicSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X1>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X2:
				HyperbolicSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X2>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X3:
				HyperbolicSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X3>(cState, lState, rState, box_r_x1, 1);
				break;
		}
	} else {
		amrex::Abort("Invalid reconstruction order specified! Supported orders: 1 (constant), 3 (PPM), 4 (xPPM).");
	}
}

template <typename problem_t>
void VectorSystem<problem_t>::SolveAdvectionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_flux_mf, double dt,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	const BL_PROFILE("VectorSystem::SolveAdvectionEqn()");
	// compute the total right-hand-side for the MOL integration

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	// loop over faces pointing in the w0-direction
	for (int w0 = 0; w0 < 3; ++w0) {
		// you have two edges on the perimeter of this face
		const int w1 = (w0 + 1) % 3; // vec_fc(w0) + vec_fc(w1)
		const int w2 = (w0 + 2) % 3; // vec_fc(w0) + vec_fc(w2)

		// direction to find the edges either side of the face. this depends on the direction the face points
		std::array<int, 3> delta_w1 = {0, 0, 0};
		std::array<int, 3> delta_w2 = {0, 0, 0};
		if (w0 == 0) {
			delta_w1[1] = 1;
			delta_w2[2] = 1;
		} else if (w0 == 1) {
			delta_w1[2] = 1;
			delta_w2[0] = 1;
		} else if (w0 == 2) {
			delta_w1[0] = 1;
			delta_w2[1] = 1;
		}

		auto const dx1 = dx[w1];
		auto const dx2 = dx[w2];
		auto const ec_flux_w1 = ec_flux_mf[w1].const_arrays();
		auto const ec_flux_w2 = ec_flux_mf[w2].const_arrays();
		auto const fc_consVarOld = fc_consVarOld_mf[w0].const_arrays();
		auto fc_consVarNew = fc_consVarNew_mf[w0].arrays();

		amrex::ParallelFor(fc_consVarNew_mf[w0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// the ec fluxes sit in the opposite fc directions relative to the face
			const double flux_w1_m = ec_flux_w1[bx](i, j, k);
			const double flux_w2_m = ec_flux_w2[bx](i, j, k);
			const double flux_w1_p = ec_flux_w1[bx](i + delta_w2[0], j + delta_w2[1], k + delta_w2[2]);
			const double flux_w2_p = ec_flux_w2[bx](i + delta_w1[0], j + delta_w1[1], k + delta_w1[2]);
			const double dv_dt = (dx1 * (flux_w1_m - flux_w1_p) + dx2 * (flux_w2_p - flux_w2_m)) / (dx1 * dx2);
			
			fc_consVarNew[bx](i, j, k, vector_index) = fc_consVarOld[bx](i, j, k, vector_index) + dt * dv_dt;
		});
	}
}

#endif // VECTOR_SYSTEM_HPP_
