#ifndef LATTICE_H
#define LATTICE_H

// --- Lattice.h ---
// It defines the "container"  of the simulation.  In the abstraction of the program, the Lattice allow the user to extract
// statistics on the behaviour of the spin, to perform cluster analysis and more. From a conceptual perspective it is the
// battlefield of the simulation.

#include <iostream>
#include "Lattice_Kernels.h"

class Lattice
{
	public:
		Lattice(int seed)
		:
		seed(seed)
		{
			// launches the kernel with a given seed
			int N = 1000;
			cudaMalloc(&grid, N * sizeof(float));
			test_kernel<<<1,N>>>(grid,N,seed);

			// retrieve data from the GPU
			float * local_grid = (float*) malloc( N * sizeof(float));

			// copy data from the gpu (grid) into the memory allocated in cpu (local_grid)
			// the number of bytes to pass is N times the size of a float the "direction"
			// is from Device (GPU) to Host (CPU)
			cudaMemcpy(local_grid, grid, N * sizeof(float), cudaMemcpyDeviceToHost);
			for(int i = 0; i < N;i++)
			{
				std::cout << "grid[" << i << "] = " << local_grid[i] << std::endl;
			}
		}

		// changes the order parameter
		void set_p(float _p)
		{
			p = _p;
		}


		// produces a visualization
		void visualize()
		{
			// TODO:  visualization function
		}

	private:
		// metadata of the simulation
			float p;
			int seed;

		// data of the simulation
			float * grid;
};

#endif
