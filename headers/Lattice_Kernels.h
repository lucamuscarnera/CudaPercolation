// --- Lattice_Kernels.h ---
// Contains the kernel used by the Lattice class


#ifndef LATTICE_KERNELS_H
#define LATTICE_KERNELS_H

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void lattice_populate(float * data, int N, int seed)
{
	// declare the state
	curandState state;

	// retrive the index
	int i = (threadIdx.x + blockIdx.x * blockDim.x) + N * (threadIdx.y + blockIdx.y * blockDim.y) ;

	// if the index  is valid
	if(i < (N*N))
	{
		// initialize the state
		curand_init(seed, i, 0 , &state);
		data[i] = curand_uniform(&state);
	}
}

#endif
