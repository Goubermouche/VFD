#ifndef DFSPH_SIMULATION_H
#define DFSPH_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Utility/Sampler/ParticleSampler.h"

namespace fe {

	struct DFSPHSimulationDescription {

	};

	/// <summary>
	/// SPH simulation wrapper
	/// </summary>
	class DFSPHSimulation : public RefCounted
	{
	public:
		DFSPHSimulation(const DFSPHSimulationDescription& desc);
		~DFSPHSimulation();

		void OnUpdate();
	public:
		bool paused = false;
	private:
	};
}

#endif // !DFSPH_SIMULATION_H