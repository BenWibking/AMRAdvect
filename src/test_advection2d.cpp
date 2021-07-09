//==============================================================================
// AMRAdvection
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include <csignal>
#include <limits>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_BoxArray.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"

#include "AdvectionSimulation.hpp"
#include "test_advection2d.hpp"

struct SquareProblem {
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
exactSolutionAtIndex(int i, int j, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
	amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
	amrex::Real const x0 = prob_lo[0] + amrex::Real(0.5) * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + amrex::Real(0.5) * (prob_hi[1] - prob_lo[1]);
	amrex::Real rho = 0.;

	if ((std::abs(x - x0) < 0.1) && (std::abs(y - y0) < 0.1)) {
		rho = 1.;
	}
	return rho;
}

template <> void AdvectionSimulation<SquareProblem>::setInitialConditionsAtLevel(int level)
{
	auto prob_lo = geom[level].ProbLoArray();
	auto prob_hi = geom[level].ProbHiArray();
	auto dx = geom[level].CellSizeArray();

	for (amrex::MFIter iter(state_old_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_[level].array(iter);

		amrex::ParallelFor(
		    indexRange, ncomp_, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
			    state(i, j, k, n) = exactSolutionAtIndex(i, j, prob_lo, prob_hi, dx);
		    });
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

template <>
void AdvectionSimulation<SquareProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags,
						  amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.5; // gradient refinement threshold
	const amrex::Real rho_min = 0.1;       // minimum rho for refinement

	for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			int const n = 0;
			amrex::Real const rho = state(i, j, k, n);

			amrex::Real const del_x = std::max(std::abs(state(i + 1, j, k, n) - rho),
							   std::abs(rho - state(i - 1, j, k, n)));
			amrex::Real const del_y = std::max(std::abs(state(i, j + 1, k, n) - rho),
							   std::abs(rho - state(i, j - 1, k, n)));
			amrex::Real const gradient_indicator =
			    std::max(del_x, del_y) / std::max(rho, rho_min);

			if (gradient_indicator > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// check that we are in strict IEEE 754 mode
	// (If we are, then the results should be symmetric [about the diagonal of the grid] not
	// only to machine epsilon but to every last digit! BUT not necessarily true when refinement
	// is enabled.)
	static_assert(std::numeric_limits<double>::is_iec559);

	// Problem parameters
	const int nx = 100;
	const double Lx = 1.0;
	const double advection_velocity = 1.0; // same for x- and y- directions
	const double CFL_number = 0.4;
	const double max_time = 1.0;
	const int max_timesteps = 1e4;
	const int nvars = 1; // only density

	amrex::IntVect gridDims{AMREX_D_DECL(nx, nx, 4)};

	amrex::RealBox boxSize{{AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
			       {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Lx), amrex::Real(1.0))}};

	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
			boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	AdvectionSimulation<SquareProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = 1;   // for debugging
	sim.checkpointInterval_ = 1; // for debugging

	sim.advectionVx_ = advection_velocity;
	sim.advectionVy_ = advection_velocity;

	// set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	int status = 0;
	return status;
}
