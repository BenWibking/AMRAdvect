# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AMRAdvect is a demonstration advection solver using AMReX that showcases flux register implementations for adaptive mesh refinement (AMR). It compares `amrex::FluxRegister` and `amrex::YAFluxRegister` for flux correction at coarse-fine interfaces.

## Build Commands

```bash
# Build from scratch
mkdir build && cd build
cmake ..
make -j  # Always use parallel builds

# Switch between flux register types
cmake -DAMRADVECT_USE_YAFLUXREGISTER=ON ..
make -j
```

## Test Commands

```bash
# Run tests from build directory
cd build
ctest -V

# Run simulation directly
./src/test_advection2d ../default.in

# Run with custom parameters
./src/test_advection2d your_params.in
```

## Architecture Overview

The codebase uses a template-based design where physics problems inherit from base simulation classes:

1. **Core Framework**: `simulation.hpp` provides the AMR simulation base class that handles mesh management, time stepping, and I/O
2. **Physics Layer**: `AdvectionSimulation.hpp` specializes the framework for advection problems
3. **Problem Setup**: Template parameter classes (e.g., `SquareProblem` in `test_advection2d.cpp`) define initial conditions and problem-specific behavior
4. **Numerical Methods**: 
   - PPM reconstruction in `hyperbolic_system.hpp`
   - RK2-SSP time integration
   - Flux register handling for AMR conservation

## Key Development Patterns

- **Adding New Test Problems**: Create a new problem class with `static void setInitialConditions()` and instantiate `AdvectionSimulation<YourProblem>`
- **Modifying Physics**: Changes to the advection solver go in `linear_advection.hpp` and `AdvectionSimulation.hpp`
- **AMR Behavior**: Flux correction logic is in `simulation.hpp` methods like `advanceHydroAtLevel()` and `fillpatch()`
- **Parameter Files**: Use AMReX ParmParse format (see `default.in` for examples)

## GPU Development

The code supports CUDA execution. Key considerations:
- Use `amrex::ParallelFor` for GPU kernels
- Ensure all physics functions are `AMREX_GPU_DEVICE` qualified
- Test with `scripts/gpu_wrapper.sh` for multi-GPU runs

## Common Issues

- If output directories don't exist, create them before running
- The `tests/` directory referenced in CMakeLists.txt doesn't exist - tests run from build directory
- For HPC runs, use the PBS script in `scripts/advect-1node.pbs`