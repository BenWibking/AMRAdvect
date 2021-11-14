#ifndef TEST_ADVECTION_HPP_ // NOLINT
#define TEST_ADVECTION_HPP_
//==============================================================================
// AMRAdvection
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

// external headers
#include "AMReX_Geometry.H"
#include "AMReX_LO_BCTYPES.H"
#include "AMReX_MLMG.H"
#include "AMReX_MLPoisson.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>

// internal headers

// function definitions
template <typename F> void fillBoundaryCells(amrex::Geometry geom, amrex::MultiFab &phi, F &&user_f);
auto problem_main() -> int;

#endif // TEST_ADVECTION_HPP_
