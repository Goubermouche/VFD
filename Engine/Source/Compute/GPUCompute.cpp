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

		s_Initialized = deviceCount > 0;

		if (s_Initialized) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0); // Choose the first device for now
			s_DeviceInfo.Name = prop.name;
			s_DeviceInfo.ClockRate = prop.clockRate;
			s_DeviceInfo.GlobalMemory = prop.totalGlobalMem;
		}
	
		if (s_Initialized) {
			LOG("GPU compute initialized successfully (device: " + s_DeviceInfo.Name + ")", "compute", ConsoleColor::Purple);
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