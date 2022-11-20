#include "pch.h"
#include "SPHSimulationComponent.h"

namespace vfd {
	SPHSimulationComponent::SPHSimulationComponent(const SPHSimulationDescription& description)
		: Handle(Ref<SPHSimulation>::Create(description))
	{}

	SPHSimulationComponent::SPHSimulationComponent(Ref<SPHSimulation> simulation)
		: Handle(simulation)
	{}
}