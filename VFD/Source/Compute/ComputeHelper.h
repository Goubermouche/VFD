#ifndef COMPUTE_HELPER_H
#define COMPUTE_HELPER_H

#include <thrust/device_vector.h>

namespace vfd
{
	class ComputeHelper
	{
	public:
		static void MemcpyHostToDevice(const void* host, void* device, const size_t size);

		template<typename T>
		static void MemcpyHostToDevice(T* host, T* device,const size_t elements)
		{
			MemcpyHostToDevice(static_cast<const void*>(host), static_cast<void*>(device), elements * sizeof(T));
		}

		static void MemcpyDeviceToHost(const void* device, void* host, const size_t size);

		template<typename T>
		static void MemcpyDeviceToHost(T* device, T* host, const size_t elements)
		{
			MemcpyDeviceToHost(static_cast<const void*>(device), static_cast<void*>(host), elements * sizeof(T));
		}

		template<typename T>
		static size_t GetSizeInBytes(const thrust::device_vector<T>& vector)
		{
			return sizeof(T) * vector.size();
		}

		template<typename T>
		static T* GetPointer(thrust::device_vector<T>& vector)
		{
			return thrust::raw_pointer_cast(&vector[0]);
		}

		static void GetThreadBlocks(const unsigned int elementCount, const unsigned int alignment, unsigned int& blockCount, unsigned int& threadCount);
	};
}

#endif // !CUDA_HELPER_H