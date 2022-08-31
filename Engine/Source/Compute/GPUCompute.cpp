#include "pch.h"
#include "GPUCompute.h"
#include "Compute/Utility/CUDA/cutil.h"


namespace fe {
	DeviceInfo GPUCompute::s_DeviceInfo;
	bool GPUCompute::s_Initialized = false;

	void GPUCompute::Init()
	{
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);

		// s_Initialized = InitCUDA(&s_DeviceInfo);
		// TODO: this is a temporary solution, update this later
		s_Initialized = deviceCount > 0;

		if (s_Initialized) {
			LOG("GPU compute initialized successfully", "compute", ConsoleColor::Purple);
		}
		else {
			ERR("failed to initialized GPU compute!", "compute");
		}

	}

	void GPUCompute::Shutdown()
	{
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			ASSERT(false, "failed to shutdown CUDA!");
		}
	}
}