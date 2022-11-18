#ifndef DFSPH_SIMULATION_COMPONENT_H
#define DFSPH_SIMULATION_COMPONENT_H

#include "Simulation/DFSPH/DFSPHSimulation.h"

namespace vfd {
	struct DFSPHSimulationComponent
	{
		Ref<DFSPHSimulation> Handle;

		DFSPHSimulationComponent() = default;
		DFSPHSimulationComponent(const DFSPHSimulationComponent& other) = default;
		DFSPHSimulationComponent(const DFSPHSimulationDescription& description);
		DFSPHSimulationComponent(Ref<DFSPHSimulation> simulation);
	};
}

#endif // !DFSPH_SIMULATION_COMPONENT_H