#include "cuda_helper.h"

namespace vfdcu {
	void CudaHelper::GetThreadBlocks(unsigned int numberOfElements, unsigned int alignment, /*out*/ unsigned int& numberOfThreadBlocks, /*out*/ unsigned int& numberOfThreads)
	{
		numberOfThreads = (numberOfElements / alignment) * alignment;
		numberOfThreadBlocks = (numberOfElements / alignment);
		if (numberOfElements % alignment != 0)
		{
			numberOfThreads += alignment;
			numberOfThreadBlocks++;
		}
	}

	void CudaHelper::MemcpyHostToDevice(void* host, void* device, size_t size)
	{
		cudaError_t cudaStatus = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			// throw CUDAMemCopyException("cudaMemcpy() failed!");
		}
	}

	void CudaHelper::MemcpyDeviceToHost(void* device, void* host, size_t size)
	{
		cudaError_t cudaStatus = cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			//throw CUDAMemCopyException("cudaMemcpy() failed!");
		}
	}
}