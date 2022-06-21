#include "System.cuh"
#include "Utility/CUDARuntimeDeviceAPI.h"

#include <stdio.h>
extern "C" {
	bool InitCuda() {
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);

		if (deviceCount == 0) {
			// No device supporting CUDA have been detected
			return false;
		}

		int device;
		cudaGetDevice(&device);

		struct cudaDeviceProp props;
		cudaGetDeviceProperties(&props, device);

		if (props.major < 1 || (device == 0 && props.major == 9999 && props.minor == 9999)) {
			// There is no device supporting CUDA
			return false;
		}

		printf("   Device name: %s\n", props.name);
		printf("   Memory Clock Rate: %d MHz\n", props.memoryClockRate / 1024);
		printf("   Total global memory: %.0f MB\n", (float)(props.totalGlobalMem) / 1024.0f / 1024.0f);
		printf("   Concurrent kernels: %s\n", props.concurrentKernels ? "yes" : "no");

		cudaSetDevice(device);

		return true;
	}
}