#ifndef COMPUTE_H_
#define COMPUTE_H_

#include "Renderer/Buffers/VertexBuffer.h"
#include "GPUCompute.cuh"
#include <cuda_runtime.h>

namespace fe {
	class GPUCompute
	{
	public:
		static void Init();
		static void Shutdown();

	    static DeviceInfo GetDeviceInfo() {
			return s_DeviceInfo;
		}

		static bool GetInitState() {
			return s_Initialized;
		}
	private:
		static DeviceInfo s_DeviceInfo;
		static bool s_Initialized;
	};
}

#endif // !COMPUTE_H_