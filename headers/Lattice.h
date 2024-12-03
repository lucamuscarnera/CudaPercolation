#ifndef LATTICE_H
#define LATTICE_H

// --- Lattice.h ---
// It defines the "container"  of the simulation.  In the abstraction of the program, the Lattice allow the user to extract
// statistics on the behaviour of the spin, to perform cluster analysis and more. From a conceptual perspective it is the
// battlefield of the simulation.

class Lattice
{
	public:
		Lattice(int seed)
		:
		seed(seed)
		{
			// launches the kernel with a given seed
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
		float p;
		int seed;
};

#endif
