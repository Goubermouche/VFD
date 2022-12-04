#ifndef GPU_DFSPH_SIMULATOR_H
#define GPU_DFSPH_SIMULATOR_H

#include "DFSPHImplementation.h"
#include "Renderer/Renderer.h"

namespace vfd
{
	struct GPUDFSPHSimulationDescription
	{
		
	};

	class DFSPHImplementation;

	// Wrapper around the GPU DFSPH simulator interface
	class GPUDFSPHSimulation : public RefCounted
	{
	public:
		GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc);

		void OnUpdate();

		const Ref<VertexArray>& GetVertexArray();
	private:
		GPUDFSPHSimulationDescription m_Description;
		std::unique_ptr<DFSPHImplementation> m_Implementation;
	};
}

#endif