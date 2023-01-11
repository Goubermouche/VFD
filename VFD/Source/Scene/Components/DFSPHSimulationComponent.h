#ifndef GPU_DFSPH_SIMULATION_COMPONENT_H
#define GPU_DFSPH_SIMULATION_COMPONENT_H

#include "Simulation/GPUDFSPH/DFSPHSimulator.h"

namespace vfd {
	struct DFSPHSimulationComponent
	{
		Ref<DFSPHSimulation> Handle;

		DFSPHSimulationComponent() = default;
		DFSPHSimulationComponent(const DFSPHSimulationComponent& other) = default;
		DFSPHSimulationComponent(DFSPHSimulationDescription& description);
		DFSPHSimulationComponent(Ref<DFSPHSimulation> simulation);
	};
}

#endif // !GPU_DFSPH_SIMULATION_COMPONENT_H