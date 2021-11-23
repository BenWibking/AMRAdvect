#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
#./build/src/test_advection2d default.in amrex.async_out=1
./build/src/test_advection2d default.in amrex.async_out=1 amrex.use_gpu_aware_mpi=1
