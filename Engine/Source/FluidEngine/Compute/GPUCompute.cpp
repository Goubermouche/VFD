#include "pch.h"
#include "GPUCompute.h"
#include "FluidEngine/Compute/Utility/CUDA/cutil.h"


namespace fe {
	DeviceInfo GPUCompute::s_DeviceInfo;
	bool GPUCompute::s_Initialized = false;

	void GPUCompute::Init()
	{
		s_Initialized = InitCUDA(&s_DeviceInfo);
		CUT_CHECK_ERROR("kernel execution failed: cuda init");

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