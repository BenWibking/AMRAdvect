//==============================================================================
// AMRPoisson
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_poisson.cpp
/// \brief Defines a test problem for a cell-centered Poisson solve.
///

#include "test_poisson.hpp"
#include "AMReX_LO_BCTYPES.H"
#include "AMReX_MLMG.H"
#include "AMReX_MLPoisson.H"
#include <AMReX.H>

auto problem_main() -> int
{
	// initialize geometry
	const int n_cell = 128;
	const int max_grid_size = 32;
	const double Lx = 1.0;

	amrex::Box domain(amrex::IntVect{AMREX_D_DECL(0, 0, 0)},
			  amrex::IntVect{AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1)});
	amrex::RealBox boxSize{{AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
			       {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Lx), amrex::Real(Lx))}};

	// set boundary conditions
	amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_lo;
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_hi;
	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
		bc_lo[i] = amrex::LinOpBCType::Periodic;
		bc_hi[i] = amrex::LinOpBCType::Periodic;
	}

	// create single-level Cartesian grids
	amrex::Geometry geom(domain, boxSize, 0, is_periodic);
	amrex::BoxArray grids(domain);
	grids.maxSize(max_grid_size);
	amrex::DistributionMapping dmap{grids};

	// MLPoisson
	amrex::MLPoisson poissoneq({geom}, {grids}, {dmap});
	poissoneq.setDomainBC(bc_lo, bc_hi);
	poissoneq.setLevelBC(0, nullptr); // set Dirichlet boundary conditions (if needed)

	// order of extrapolation to ghost cell center
	// (see
	// https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html#boundary-stencils-for-cell-centered-solvers)
	poissoneq.setMaxOrder(2);

	// MLMG
	amrex::MLMG mlmg(poissoneq);
	mlmg.setVerbose(1);
	mlmg.setBottomVerbose(0);
	mlmg.setBottomSolver(amrex::MLMG::BottomSolver::bicgstab);
	mlmg.setMaxFmgIter(1);

	const int ncomp = 1;
	const int nghost = 0;
	const int nlev = 1;
	amrex::MultiFab phi(grids, dmap, ncomp, nghost);
	amrex::MultiFab rhs(grids, dmap, ncomp, nghost);
	amrex::Vector<amrex::MultiFab *> phi_levels(nlev);
	amrex::Vector<amrex::MultiFab const *> rhs_levels(nlev);
	phi_levels[0] = &phi;
	rhs_levels[0] = &rhs;

	// initial guess for phi
	phi.setVal(0);

	// set density field to a Fourier mode of the box
	// (N.B. very slow convergence when kx=ky=1...)
	const int kx = 1;
	const int ky = 1;

	auto prob_lo = geom.ProbLoArray();
	auto dx = geom.CellSizeArray();
	for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto rho = rhs.array(mfi);
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
			rho(i, j, k) = std::sin(2.0 * M_PI * kx * x) * std::sin(2.0 * M_PI * ky * y);
		});
	}

	const amrex::Real reltol = 1.0e-13; // doesn't work below 1e-13...
	const amrex::Real abstol = 0; // unused if zero
	// L_\infty-norm residual || \phi - \phi_exact ||_{\infty}
	amrex::Real residual_linf = mlmg.solve(phi_levels, rhs_levels, reltol, abstol);

	std::cout << "Residual max norm = " << residual_linf << "\n\n";

	int status = 0;
	return status;
}
