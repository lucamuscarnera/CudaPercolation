#ifndef LATTICE_H
#define LATTICE_H

// --- Lattice.h ---
// It defines the "container"  of the simulation.  In the abstraction of the program, the Lattice allow the user to extract
// statistics on the behaviour of the spin, to perform cluster analysis and more. From a conceptual perspective it is the
// battlefield of the simulation.

#include <iostream>
#include "Settings.h"
#include "Lattice_Kernels.h"

class Lattice
{
	public:
		Lattice(int seed,			  // seed of the simulation
			int N,                            // number of sites (height and width equal)
			const Settings & settings)	  // constant reference to settings object, that contains information on the simulation
		:
		seed(seed)
		{
			// retrieves the maximum number of threads ina single block, and construct the partition
			int threads_one_direction = int(sqrt(settings
							     .get_GPU_settings()
							     .max_threads_per_block));

			dim3 N_threads(threads_one_direction,
				       threads_one_direction);

			dim3 N_blocks(int(N/threads_one_direction),
				      int(N/threads_one_direction));

			// initialize  the  global memory region
			cudaMalloc(&grid, (N * N) * sizeof(float));

			// launch the kernel
			lattice_populate<<<N_blocks,N_threads>>>(grid,N,seed);

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
