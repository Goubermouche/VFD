#include <string>

extern "C" {
	struct CUDADeviceInfo {
		std::string name;
		int clockRate; // hz
		int globalMemory; // bytes
		bool concurrentKernels;
		int coreCount;
	};

	bool k_Init(CUDADeviceInfo* deviceInfo);
}