#ifndef DFSPH_SIMULATION_H
#define DFSPH_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Simulation/DFSPH/CompactNSearch.h"

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
		NeighborhoodSearch* m_NeighbourHoodSearch;
	};
}

#endif // !DFSPH_SIMULATION_H