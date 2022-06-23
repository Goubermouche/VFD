#ifndef COMPUTE_H_
#define COMPUTE_H_

#include "FluidEngine/Renderer/Buffers/VertexBuffer.h"
#include "GPUCompute.cuh"
#include <cuda_runtime.h>

namespace fe {
	class GPUComputeResource : public RefCounted {
	public:
		GPUComputeResource() = default;
		~GPUComputeResource() {
			cudaGraphicsUnregisterResource(m_Resource);
		}

		cudaGraphicsResource** Get() {
			return &m_Resource;
		}

		void Unregister() {
			cudaGraphicsUnregisterResource(m_Resource);
		}

		void Map(void** dptr) {
			cudaGraphicsMapResources(1, &m_Resource, 0);
			size_t bufferSize; // bytes
			cudaGraphicsResourceGetMappedPointer(dptr, &bufferSize, m_Resource);
		}

		void Unmap() {
			cudaGraphicsUnmapResources(1, &m_Resource, 0);
		}

	private:
		cudaGraphicsResource* m_Resource;
	};

	class GPUCompute
	{
	public:
		static void Init();

	    static DeviceInfo GetDeviceInfo() {
			return s_DeviceInfo;
		}

		static bool GetInitState() {
			return s_Initialized;
		}

		static void RegisterBuffer(Ref<GPUComputeResource> resource, Ref<VertexBuffer> buffer, cudaGraphicsMapFlags flags = cudaGraphicsMapFlagsNone);
	private:
		static DeviceInfo s_DeviceInfo;
		static bool s_Initialized;
	};
}

#endif // !COMPUTE_H_