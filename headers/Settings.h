#ifndef SETTINGS_H
#define SETTINGS_H

// --- Settings.h ---
// Wrapper class for containing information on the hardware, on the simulation etc. During initialization of a Settings object
// some fields (settings) are chosen in order to be optimal. Every class (Lattice,...) has to be initialized with Settings 
// object.

// TODO: csv/json parser

#include <cuda_runtime.h>

class Settings
{
	public:
		Settings()
		{
			// fill the field of the maximum number of threads per block
			cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
		}
	private:
		struct GPU
		{
			unsigned int max_threads_per_block;
		};
};

#endif
