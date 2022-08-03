#ifndef CUDA_KERNEL_UTILITY_CUH_
#define CUDA_KERNEL_UTILITY_CUH_

namespace fe {
	extern "C" {
		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount);
	}
}

#endif // !CUDA_KERNEL_UTILITY_CUH_