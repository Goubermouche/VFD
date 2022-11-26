#ifndef GPUSDF_CU
#define GPUSDF_CU

#include "pch.h"
#include "GPUSDF.cuh"
#include <cuda_gl_interop.h>

namespace vfd {
	extern "C" {
		static __global__  void TestKernel(Arr<int> arr) {
			const uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

			arr[index] *= 10;
			printf("%d\n", arr[index]);
		}

		void TestCUDA()
		{
			Arr<int> arr(4);
			arr.PushBack(1);
			arr.PushBack(2);
			arr.PushBack(3);
			arr.PushBack(4);

			ERR("HOST:");
			for (int val : arr) {
				std::cout << val << std::endl;
			}

			ERR("MOVING TO DEVICE...");
			arr.MoveToDevice();

			ERR("RUNNING KERNELS...");
			ERR("DEVICE:");
			TestKernel << < 1, 4 >> > (arr);

			arr.MoveToHost();
			ERR("HOST:");

			for (int val : arr) {
				std::cout << val << std::endl;
			}

			COMPUTE_CHECK("Kernel execution failed");
			COMPUTE_SAFE(cudaDeviceSynchronize());
		}
	}
}

#endif // !GPUSDF_CU
