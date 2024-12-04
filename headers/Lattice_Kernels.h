// --- Lattice_Kernels.h ---
// Contains the kernel used by the Lattice class


#ifndef LATTICE_KERNELS_H
#define LATTICE_KERNELS_H

__global__ void test_kernel(float * data, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < N)
	{
		data[i] = 1.;
	}
}

#endif
