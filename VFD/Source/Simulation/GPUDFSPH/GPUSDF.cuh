#ifndef GPU_SDF_CUH
#define GPU_SDF_CUH

#include "host_defines.h"
#include "cuda_runtime.h"

namespace vfd {
	template<class T>
	struct Iterator {
		__host__ __device__ Iterator() {}
		__host__ __device__ Iterator(T* ptr) 
		: m_Ptr(ptr) {}

		__host__ __device__ bool operator==(const Iterator& rhs) const {
			return m_Ptr == rhs.m_Ptr;
		}
		__host__ __device__ bool operator!=(const Iterator& rhs) const {
			return !(*this == rhs);
		}
		__host__ __device__ T operator*() const {
			return *m_Ptr;
		}
		__host__ __device__ Iterator& operator++()	{
			++m_Ptr;
			return *this;
		}
		__host__ __device__ Iterator operator++(int) {
			Iterator temp(*this);
			++* this;
			return temp;
		}
	private:
		T* m_Ptr = nullptr;
	};

	// TODO: 
	// - atomic operations (push, pop etc.)
	// - device-side push
	// - reference iterator

	/// <summary>
	/// Host/device dynamic array, similar to the stl vector.
	/// </summary>
	/// <typeparam name="T">Data type.</typeparam>
	template<class T>
	struct Arr {
		__host__ __device__ Arr() {
			// printf("init vector\n");
		}
		__host__ __device__ Arr(unsigned int size)
		: m_Capacity(size), m_Size(0), m_Data(new T[size]) {}

		// Accessors 
		__host__ __device__ unsigned int PushBack(T data) {
			if (m_Size == m_Capacity) {
				T* temp = new T[m_Capacity * 2];
				memcpy(temp, m_Data, sizeof(*m_Data) * m_Capacity);
				delete[] m_Data;
				m_Data = temp;
				m_Capacity *= 2;
			}

			m_Data[m_Size++] = data;
			return m_Size;
		}
		__host__ __device__ T PopBack() {
			return m_Data[m_Size-- - 1];
		}
		__host__ __device__ T& At(unsigned int index) {
			if (index >= m_Capacity) {
				printf("Error: index out of range!\n");
				exit(0);
			}

			return *(m_Data + index);
		}

		/// <summary>
		/// Moves the instance over to device memory and frees host-side memory.
		/// </summary>
		__host__ void MoveToDevice() {
			T* device; // Temporary memory buffer

			// Copy host memory over to the device
			COMPUTE_SAFE(cudaMalloc(&(device), m_Capacity * sizeof(T)));
			COMPUTE_SAFE(cudaMemcpy(device, m_Data, m_Capacity * sizeof(T), cudaMemcpyHostToDevice));

			delete[] m_Data; // Free host-side memory since we've copied it to the GPU
			m_Data = device;
		}

		/// <summary>
		/// Moves the instance data over to a device-side entity, host-side data is preserved and the current entity is unchanged.
		/// </summary>
		/// <param name="device">Device-side instance that the host-side instance data will be copied over to.</param>
		__host__ void CopyToDevice(Arr<T>& device) {
			device.m_Size = m_Size;
			device.m_Capacity = m_Capacity;

			// Copy host memory over to the device
			COMPUTE_SAFE(cudaMalloc(&(device.m_Data), m_Capacity * sizeof(T)));
			COMPUTE_SAFE(cudaMemcpy(device.m_Data, m_Data, m_Capacity * sizeof(T), cudaMemcpyHostToDevice));
		}

		/// <summary>
		/// Moves the instance over to host memory and frees device-side memory.
		/// </summary>
		__host__ __device__ void MoveToHost() {
			T* host = new T[m_Capacity]; // Temporary memory buffer

			// Copy device memory over to the hostc
			COMPUTE_SAFE(cudaMemcpy(host, m_Data, m_Capacity * sizeof(T), cudaMemcpyDeviceToHost));
			COMPUTE_SAFE(cudaFree(m_Data)); // Free device-side memory since we've copied it to the CPU

			m_Data = host;
		}

		/// <summary>
		/// Moves the instance data over to a host-side entity, device-side data is preserved and the current entity is unchanged.
		/// </summary>
		/// <param name="host">Host-side instance that the device-side instance data will be copied over to.</param>
		__host__ __device__ void CopyToHost(Arr<T>& host) {
			delete[] host.m_Data;

			host.m_Size = m_Size;
			host.m_Capacity = m_Capacity;
			host.m_Data = new T[m_Capacity];

			// Copy device memory over to the host
			COMPUTE_SAFE(cudaMemcpy(host.m_Data, m_Data, host.m_Capacity * sizeof(T), cudaMemcpyDeviceToHost));
		}

		// Getters
		__host__ __device__ unsigned int GetSize() {
			return m_Size;
		}
		__host__ __device__ unsigned int GetCapacity() {
			return m_Capacity;
		}
		__host__ __device__ T* GetData() {
			return m_Data;
		}

		// Overloads
		__host__ __device__ T& operator[](unsigned int index) {
			return At(index);
		}
		// Iterator accessors have to be lowercase so that the compiler picks them up
		__host__ __device__ Iterator<T> begin() const {
			return Iterator<T>(m_Data);
		}
		__host__ __device__ Iterator<T> end() const {
			return Iterator<T>(m_Data + m_Capacity);
		}
		__host__ __device__ Iterator<T> begin() {
			return Iterator<T>(m_Data);
		}
		__host__ __device__ Iterator<T> end() {
			return Iterator<T>(m_Data + m_Capacity);
		}
	private:
		T* m_Data = nullptr;         // Array contents

		unsigned int m_Size = 0;     // Count of m_Data elements
		unsigned int m_Capacity = 0; // Current max capacity of m_Data
	};

	// TEMP
	extern "C" {
		void TestCUDA();
	}
}

#endif // !GPU_SDF_CUH