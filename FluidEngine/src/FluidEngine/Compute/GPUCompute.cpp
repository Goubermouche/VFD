#include "pch.h"
#include "GPUCompute.h"

#include "FluidEngine/Compute/Utility/CUDAGLInterop.h"

namespace fe {
	DeviceInfo GPUCompute::s_DeviceInfo;
	bool GPUCompute::s_Initialized = false;

	void GPUCompute::Init()
	{
		s_Initialized = k_Init(&s_DeviceInfo);

		if (s_Initialized) {
			LOG("GPU compute initialized successfully");
		}
		else {
			ERR("failed to initialized GPU compute!");
		}
	}

	void GPUCompute::RegisterBuffer(Ref<GPUComputeResource> resource, Ref<VertexBuffer> buffer, cudaGraphicsMapFlags flags)
	{
		cudaGraphicsGLRegisterBuffer(resource->Get(), buffer->GetRendererID(), flags);
	}
}