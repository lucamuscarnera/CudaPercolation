#ifndef SETTINGS_H
#define SETTINGS_H

// --- Settings.h ---
// Wrapper class for containing information on the hardware, on the simulation etc. During initialization of a Settings object
// some fields (settings) are chosen in order to be optimal. Every class (Lattice,...) has to be initialized with Settings 
// object.

// TODO: csv/json parser
#include <iostream>
#include <cuda_runtime.h>

class Settings
{
	public:
		Settings()
		{
			// fill the field of the maximum number of threads per block
			cudaDeviceGetAttribute(&(GPU.max_threads_per_block), cudaDevAttrMaxThreadsPerBlock, 0);
		}

		// shows summary of settings
		void summary()
		{
			std::cout << std::endl;
			std::cout << "========== Settings ==========" << std::endl;
			std::cout << "---- GPU ---------------------" << std::endl;
			std::cout << "Max threads per block = " << GPU.max_threads_per_block << std::endl;
			std::cout << "========== ======== ==========" << std::endl;
			std::cout << std::endl;
		}
	private:
		struct
		{
			int max_threads_per_block;
		} GPU;
};

#endif
