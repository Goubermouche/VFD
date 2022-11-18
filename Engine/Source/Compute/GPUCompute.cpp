#include "pch.h"
#include "GPUCompute.h"
#include "Compute/Utility/CUDA/cutil.h"
#include "Utility/FileSystem.h"

namespace fe {
	DeviceInfo GPUCompute::s_DeviceInfo;
	bool GPUCompute::s_Initialized = false;

	void GPUCompute::Init()
	{
		int deviceCount;
		s_Initialized = cudaGetDeviceCount(&deviceCount) == cudaSuccess;

		if (s_Initialized) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0); // Choose the first device for now
			s_DeviceInfo.Name = prop.name;
			s_DeviceInfo.ClockRate = prop.clockRate;
			s_DeviceInfo.GlobalMemory = prop.totalGlobalMem;

			std::cout << "Device: " << s_DeviceInfo.Name << '\n'
				      << "Device memory: " << FormatFileSize(s_DeviceInfo.GlobalMemory) << '\n';
		}
		else {
			std::cout << "Device: no CUDA-capable device was found\n";
		}
	}

	void GPUCompute::Shutdown()
	{
		if (s_Initialized) {
			if (cudaDeviceReset() != cudaSuccess) {
				ASSERT("failed to shutdown CUDA!");
			}
		}
	}
}