#ifndef LATTICE_H
#define LATTICE_H

// --- Lattice.h ---
// It defines the "container"  of the simulation.  In the abstraction of the program, the Lattice allow the user to extract
// statistics on the behaviour of the spin, to perform cluster analysis and more. From a conceptual perspective it is the
// battlefield of the simulation.

#include <iostream>
#include "Settings.h"
#include "Lattice_Kernels.h"
#define _DEBUG_SHOW_AVERAGE_SPIN



#include <fstream>

void dumpArrayToCSV(const unsigned int* array, size_t size, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < size; ++i) {
        file << array[i];  // Write the value to the file

        if (i < size - 1) {
            file << ",";    // Add a comma except after the last element
        }
    }

    file << "\n";  // End with a newline
    file.close();   // Close the file

    std::cout << "Data successfully written to " << filename << std::endl;
}



class Lattice
{
	public:
		Lattice(int seed,			  // seed of the simulation
			int N,                            // number of sites (height and width equal)
			const Settings & settings)	  // constant reference to settings object, that contains information on the simulation
		:
		N(N),
		seed(seed),
		settings(settings)
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
			cudaMalloc(&threshold_grid, (N * N) * sizeof(float));

			// launch the kernel for populating the sites
			lattice_populate<<<N_blocks,N_threads>>>(threshold_grid,N,seed);
			cudaDeviceSynchronize();
		}

		// run a simulation
		void observe(float p)
		{
			// retrieves the maximum number of threads ina single block, and construct the partition
			int threads_one_direction = int(sqrt(settings
							     .get_GPU_settings()
							     .max_threads_per_block));

			dim3 N_threads(threads_one_direction,
				       threads_one_direction);

			dim3 N_blocks(int(N/threads_one_direction),
				      int(N/threads_one_direction));


			// call the observation kernel
			cudaMalloc(&state_grid, (N * N) * sizeof(unsigned char));
			lattice_observe<<< N_blocks, N_threads >>>(threshold_grid,state_grid, N, p);
			cudaDeviceSynchronize();

			// call the id assigning kernel
			unsigned int * id_grid;
			cudaMalloc(&id_grid, (N*N) * sizeof(unsigned int));
			lattice_assign_id<<< N_blocks, N_threads >>>(state_grid, id_grid, N);

			// diffusion process
			for(int i = 0 ; i < 2 * N;i++)
				lattice_id_diffusion<<< N_blocks, N_threads >>>(id_grid, N);
				cudaDeviceSynchronize();

			//  save in a file
                        unsigned int * host_id_grid = (unsigned int *) malloc(sizeof(unsigned int) * (N*N));
                        cudaMemcpy(host_id_grid,
                                   id_grid,
                                   (N*N) * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost);
			dumpArrayToCSV(host_id_grid, N  * N , "dump.csv");
                        cudaFree(state_grid);




			#ifdef DEBUG_SHOW_AVERAGE_SPIN
			unsigned char * host_state_grid = (unsigned char *) malloc(sizeof(unsigned char) * (N*N));
			cudaMemcpy(host_state_grid,
				   state_grid,
				   (N*N) * sizeof(unsigned char),
				   cudaMemcpyDeviceToHost);
			int accesi = 0;
			for(int i = 0;i < N*N;i++)
				accesi += ((int)(host_state_grid[i] == 1));
			std:: cout << "the average spin is " << ((float) accesi)/((float) (N*N)) << std::endl;
			cudaFree(state_grid);
       			#endif

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
		// settings
			const Settings & settings;
		// metadata of the simulation
			float p;
			int seed;
			int N;
		// data of the simulation
			float * threshold_grid;
			unsigned char * state_grid;
};

#endif
