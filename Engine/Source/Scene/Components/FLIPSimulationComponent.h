#ifndef FLIP_SIMULATION_COMPONENT_H
#define FLIP_SIMULATION_COMPONENT_H

#include "Simulation/FLIP/FLIPSimulation.h"

namespace fe {
	struct FLIPSimulationComponent
	{
		Ref<FLIPSimulation> Handle;

		FLIPSimulationComponent() = default;
		FLIPSimulationComponent(const FLIPSimulationComponent& other) = default;
		FLIPSimulationComponent(const FLIPSimulationDescription& description)
			: Handle(Ref<FLIPSimulation>::Create(description))
		{}
		FLIPSimulationComponent(Ref<FLIPSimulation> simulation)
			: Handle(simulation)
		{}

		// TODO: Add serialization methods
	};
}

#endif // !FLIP_SIMULATION_COMPONENT_H