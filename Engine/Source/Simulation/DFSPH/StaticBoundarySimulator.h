#ifndef STATIC_BOUNDARY_SIMULATOR_H
#define STATIC_BOUNDARY_SIMULATOR_H

#include "DFSPHSimulation.h"

namespace fe {
	class DFSPHSimulation;
	class StaticBoundarySimulator
	{
	public:
		StaticBoundarySimulator(DFSPHSimulation* base);
	private:
		DFSPHSimulation* m_base;
	};
}
#endif // !STATIC_BOUNDARY_SIMULATOR_H