#include <iostream>
#include "headers/Lattice.h"

int main()
{
	Settings base_settings;
	base_settings.summary();
	Lattice lattice(42, 256, base_settings);
	// run the simulation
	lattice.observe(0.4);
}
