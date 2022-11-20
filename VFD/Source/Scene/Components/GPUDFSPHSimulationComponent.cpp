#include "pch.h"
#include "GPUDFSPHSimulationComponent.h"

namespace vfd {
	GPUDFSPHSimulationComponent::GPUDFSPHSimulationComponent(const GPUDFSPHSimulationDescription& description)
		: Handle(Ref<GPUDFSPHSimulation>::Create(description))
	{}

	GPUDFSPHSimulationComponent::GPUDFSPHSimulationComponent(Ref<GPUDFSPHSimulation> simulation)
		: Handle(simulation)
	{}
}