#ifndef CUDA_KERNEL_UTILITY_CUH_
#define CUDA_KERNEL_UTILITY_CUH_

#include <glm/glm.hpp>
#include "Compute/Utility/CUDA/cutil_math.h"

namespace fe {
	void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount);
	void ComputeGridSize(const glm::ivec2& n, const glm::ivec2& blockSize, dim3& blockCount, dim3& threadCount);
	void ComputeGridSize(const glm::ivec3& n, const glm::ivec3& blockSize, dim3& blockCount, dim3& threadCount);
}

#endif // !CUDA_KERNEL_UTILITY_CUH_