#ifndef DFSPH_SIMULATION_COMPONENT_H
#define DFSPH_SIMULATION_COMPONENT_H

#include "pch.h"
#include "Simulation/DFSPH/DFSPHSimulation.h"

namespace fe {
	struct DFSPHSimulationComponent
	{
		Ref<DFSPHSimulation> Handle;

		DFSPHSimulationComponent() = default;
		DFSPHSimulationComponent(const DFSPHSimulationComponent& other) = default;
		DFSPHSimulationComponent(const DFSPHSimulationDescription& description)
			: Handle(Ref<DFSPHSimulation>::Create(description))
		{}
		DFSPHSimulationComponent(Ref<DFSPHSimulation> simulation)
			: Handle(simulation)
		{}
	};
}

#endif // !DFSPH_SIMULATION_COMPONENT_H