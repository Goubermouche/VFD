#include "pch.h"
#include "ComputeHelper.h"

namespace vfd
{
	void ComputeHelper::MemcpyHostToDevice(const void* host, void* device, const size_t size)
	{
		COMPUTE_SAFE(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
	}

	void ComputeHelper::MemcpyDeviceToHost(const void* device, void* host, const size_t size)
	{
		COMPUTE_SAFE(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
	}

	void ComputeHelper::GetThreadBlocks(const unsigned int elementCount, const unsigned int alignment, /*out*/ unsigned int& blockCount, /*out*/ unsigned int& threadCount)
	{
		threadCount = std::min(alignment, elementCount);

		if(elementCount % threadCount == 0) {
			blockCount = elementCount / threadCount;
		}
		else {
			blockCount = elementCount / threadCount + 1;
		}
	}
}