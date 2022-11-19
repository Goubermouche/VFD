#ifndef GPU_DFSPH_SIMULATION_COMPONENT_H
#define GPU_DFSPH_SIMULATION_COMPONENT_H

#include "Simulation/GPUDFSPH/GPUDFSPHSimulator.h"

namespace vfd {
	struct GPUDFSPHSimulationComponent
	{
		Ref<GPUDFSPHSimulation> Handle;

		GPUDFSPHSimulationComponent() = default;
		GPUDFSPHSimulationComponent(const GPUDFSPHSimulationComponent& other) = default;
		GPUDFSPHSimulationComponent(const GPUDFSPHSimulationDescription& description);
		GPUDFSPHSimulationComponent(Ref<GPUDFSPHSimulation> simulation);
	};
}

#endif // !GPU_DFSPH_SIMULATION_COMPONENT_H