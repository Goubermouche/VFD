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

	void GPUCompute::UnregisterResource(Ref<GPUComputeResource> resource)
	{
		cudaGraphicsUnregisterResource(*resource->Get());
	}

	void GPUCompute::RegisterBuffer(Ref<GPUComputeResource> resource, Ref<VertexBuffer> buffer, cudaGraphicsMapFlags flags)
	{
		cudaGraphicsGLRegisterBuffer(resource->Get(), buffer->GetRendererID(), flags);
	}

	void GPUCompute::MapResource(Ref<GPUComputeResource> resource, void** data)
	{
		cudaGraphicsMapResources(1, resource->Get(), 0);
		size_t bufferSize; // bytes
		cudaGraphicsResourceGetMappedPointer(data, &bufferSize, *resource->Get());
	}

	void GPUCompute::UnmapResource(Ref<GPUComputeResource> resource)
	{
		cudaGraphicsUnmapResources(1, resource->Get(), 0);
	}
}