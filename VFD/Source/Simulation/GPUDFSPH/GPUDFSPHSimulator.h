#ifndef GPU_DFSPH_SIMULATOR_H
#define GPU_DFSPH_SIMULATOR_H

#include "GPUSDF.h"

namespace vfd
{
	struct GPUDFSPHSimulationDescription
	{
		
	};

	// Wrapper around the GPU DFSPH simulator interface
	class GPUDFSPHSimulation : public RefCounted
	{
	public:
		GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc);

		void OnRender();
		void OnUpdate();
	private:
		GPUDFSPHSimulationDescription m_Description;

		Ref<GPUSDF> m_SDF;
		std::vector<glm::vec3> samples;
	};
}

#endif