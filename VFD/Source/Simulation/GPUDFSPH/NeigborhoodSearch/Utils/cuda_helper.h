#ifndef CUDA_HELPER_H 
#define CUDA_HELPER_H

#include <thrust/device_vector.h>

namespace vfdcu {
	class CudaHelper {
	public:
		static void GetThreadBlocks(unsigned int numberOfElements, unsigned int alignment, /*out*/ unsigned int& numberOfThreadBlocks, /*out*/ unsigned int& numberOfThreads);

		template<typename T>
		static T* GetPointer(thrust::device_vector<T>& vector)
		{
			return thrust::raw_pointer_cast(&vector[0]);
		}

		static void MemcpyHostToDevice(void* host, void* device, size_t size);

		template<typename T>
		static void MemcpyHostToDevice(T* host, T* device, size_t elements)
		{
			MemcpyHostToDevice((void*)host, (void*)device, elements * sizeof(T));
		}

		template<typename T>
		static size_t GetSizeInBytes(const thrust::device_vector<T>& vector)
		{
			return sizeof(T) * vector.size();
		}

		static void MemcpyDeviceToHost(void* device, void* host, size_t size);

		/** Copies data from device to host.
		*/
		template<typename T>
		static void MemcpyDeviceToHost(T* device, T* host, size_t elements)
		{
			MemcpyDeviceToHost((void*)device, (void*)host, elements * sizeof(T));
		}
	};
}

#endif // !CUDA_HELPER_H 