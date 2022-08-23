#include "pch.h"
#include "CudaKernelUtility.cuh"

#include "Compute/Utility/CUDA/cutil_math.h"

namespace fe {
	extern "C" {
		int IDivUp(int a, int b) {
			return a % b != 0 ? a / b + 1 : a / b;
		}

		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount) {
			threadCount = min(blockSize, n);
			blockCount = IDivUp(n, threadCount);
		}
	}
}