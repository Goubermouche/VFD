#include "pch.h"
#include "CUDACompute.h"


namespace fe::cuda {
	void CUDACompute::Init()
	{
		CUDADeviceInfo info;
		s_InitializedSuccessfully = k_Init(&info);

		if (s_InitializedSuccessfully) {
			s_DeviceInfo.name = info.name;
			s_DeviceInfo.clockRate = info.clockRate;
			s_DeviceInfo.globalMemory = info.globalMemory;
			s_DeviceInfo.concurrentKernels = info.concurrentKernels;
			s_DeviceInfo.coreCount = info.coreCount;

			LOG("CUDA initialized successfully");
		}
		else {
			ERR("failed to initialize CUDA!");
		}
	}
}