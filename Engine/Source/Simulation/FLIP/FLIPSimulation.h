#ifndef FLIP_SIMULATION_H
#define FLIP_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"

namespace fe {
	struct FLIPSimulationDescription {

	};

	class FLIPSimulation : public RefCounted
	{
	public:
		FLIPSimulation(const FLIPSimulationDescription& desc);
		~FLIPSimulation();
	private:
		FLIPSimulationDescription m_Description;
	};
}

#endif // !FLIP_SIMULATION_H