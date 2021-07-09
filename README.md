# AMRAdvect

This is a simple advection solver built with AMReX designed to illustrate the use of amrex::FluxRegister and amrex::YAFluxRegister for refluxing at coarse-fine interfaces.

In some situations, it appears that YAFluxRegister does not give the same answer as FluxRegister. To see this, build the code using CMake with  AMRADVECT_USE_YAFLUXREGISTER set to ON and OFF, respectively (set this in CMakeLists.txt). Run the test_advection2d executable using the provided default.in parameter file.
