# AMRAdvect

This is a simple advection solver built with AMReX designed to illustrate the use of amrex::FluxRegister and amrex::YAFluxRegister for refluxing at coarse-fine interfaces.

Build the code using CMake with  AMRADVECT_USE_YAFLUXREGISTER set to either ON or OFF in order to use YAFluxRegister or FluxRegister, respectively. Run the test_advection2d executable using the provided default.in parameter file.
