#include "pch.h"
#include "CudaKernelUtility.cuh"

namespace vfd {
	int IDivUp(int a, int b) {
		return a % b != 0 ? a / b + 1 : a / b;
	}

	void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount) {
		threadCount = min(blockSize, n);
		blockCount = IDivUp(n, threadCount);
	}

	void ComputeGridSize(const glm::ivec2& n, const glm::ivec2& blockSize, dim3& blockCount, dim3& threadCount)
	{
		blockCount = dim3(blockSize.x, blockSize.y);

		threadCount.x = (n.x + blockCount.x - 1) / blockCount.x;
		threadCount.y = (n.y + blockCount.y - 1) / blockCount.y;
	}

	void ComputeGridSize(const glm::ivec3& n, const glm::ivec3& blockSize, dim3& blockCount, dim3& threadCount)
	{
		blockCount = dim3(blockSize.x, blockSize.y);

		threadCount.x = (n.x + blockCount.x - 1) / blockCount.x;
		threadCount.y = (n.y + blockCount.y - 1) / blockCount.y;
		threadCount.z = (n.z + blockCount.z - 1) / blockCount.z;
	}
}