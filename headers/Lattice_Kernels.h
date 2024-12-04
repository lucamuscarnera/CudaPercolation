// --- Lattice_Kernels.h ---
// Contains the kernel used by the Lattice class


#ifndef LATTICE_KERNELS_H
#define LATTICE_KERNELS_H

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


// associates each site with the realization of a uniform random
// variable between 0 and 1. 
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

// Thresholds the value obtained through lattice_populate. The closer is p to 0, the lower is the threshold
// to observe a "spin up"
__global__ void lattice_observe(float * lattice_in,
				unsigned char * lattice_out,
				int N,
				float p)
{
	// retrieve the index
	int i = (threadIdx.x + blockIdx.x * blockDim.x) + N * (threadIdx.y + blockIdx.y * blockDim.y) ;

	if(i < (N*N))
	{
		if(lattice_in[i] < p)
		{
			lattice_out[i] = 1;
		}
	}
}


// Assign to every "on" site a number, such that each site has a unique id. The assignement happens populating a portion of memory
// composed of unsigned int. The kernel reads the lattice produced by  a lattice observation, and assigns
//
//                    -> 0               if the site is off
//                    -> i + 1           if the site is on
//
// Conceptually, we can think that we are constructing isolated clusters with only one site, and numbering them from 1 to N^2 + 1


__global__ void lattice_assign_id(unsigned char * lattice_in,
				  unsigned int * lattice_out,
				  int N)
{
	// retrieve index
        int i = (threadIdx.x + blockIdx.x * blockDim.x) + N * (threadIdx.y + blockIdx.y * blockDim.y) ;

	if(i < (N*N))
	{
		if(lattice_in[i] == 1)
		{
			lattice_out[i] = i + 1;
		}
	}
}

#endif
