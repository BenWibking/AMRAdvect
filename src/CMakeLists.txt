include_directories(${amrex_INCLUDE_DIRS_RET})
link_libraries(AMReX::amrex)

add_executable(test_poisson main.cpp test_poisson.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_poisson)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

include(CTest)
message(DEBUG "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
add_test(NAME Poisson2D COMMAND test_poisson Poisson2D.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
