#include "pch.h"
#include "DFSPHSimulationComponent.h"

namespace vfd {
	DFSPHSimulationComponent::DFSPHSimulationComponent(const DFSPHSimulationDescription& description)
		: Handle(Ref<DFSPHSimulation>::Create(description))
	{}

	DFSPHSimulationComponent::DFSPHSimulationComponent(Ref<DFSPHSimulation> simulation)
		: Handle(simulation)
	{}
}