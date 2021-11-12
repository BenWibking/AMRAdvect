cmake_minimum_required(VERSION 3.16)
project(AMRAdvection VERSION 1.0
  DESCRIPTION "AMR advection test with refluxing"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/extern/amrex/Tools/CMake)
include(CTest)

set(AMReX_SPACEDIM 2 CACHE STRING "" FORCE)
set(AMReX_MPI OFF CACHE BOOL "" FORCE)
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_PROBINIT OFF CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS ON CACHE BOOL "" FORCE)
set(AMReX_ASSERTIONS OFF CACHE BOOL "" FORCE)
set(AMReX_FPE OFF CACHE BOOL "" FORCE)
set(AMReX_TINY_PROFILE OFF CACHE BOOL "" FORCE)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  enable_language(CUDA)
  set(AMReX_CUDA_FASTMATH OFF CACHE BOOL "" FORCE)
  # fused MAD causes directional asymmetry
  # (unfortunately, disabling reduces GPU performance by ~30%)
  set(CMAKE_CUDA_FLAGS "--fmad=false" CACHE STRING "" FORCE)
  # Include the AMReX-provided CUDA setup module -- OBSOLETE with CMake >= 3.20
  if(CMAKE_VERSION VERSION_LESS 3.20)
    include(AMReX_SetupCUDA)
  endif(CMAKE_VERSION VERSION_LESS 3.20)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

# set which FluxRegister to use
set(AMRADVECT_USE_YAFLUXREGISTER ON)

if (AMRADVECT_USE_YAFLUXREGISTER)
  add_compile_definitions(USE_YAFLUXREGISTER)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/extern/amrex ${PROJECT_BINARY_DIR}/amrex)
add_subdirectory(${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)
