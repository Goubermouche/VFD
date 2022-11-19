#ifndef GPU_DFSPH_SIMULATOR_H
#define GPU_DFSPH_SIMULATOR_H

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

	private:
		GPUDFSPHSimulationDescription m_Description;
	};
}

#endif