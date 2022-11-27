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
		__host__ __device__ Arr(unsigned int capacity) {
			ASSERT(capacity > 0, "Array size must ");

			COMPUTE_SAFE(cudaMallocManaged(&m_Data, capacity * sizeof(T)));
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 2 * sizeof(unsigned int)));

			m_Info[0] = 0;
			m_Info[1] = capacity;
		}
		__host__ __device__ void Free() {
			COMPUTE_SAFE(cudaFree(m_Data));
			COMPUTE_SAFE(cudaFree(m_Info));
		}

		__host__ unsigned int PushBack(const T& data) {
			if (m_Info[0] == m_Info[1]) {
				T* temp = nullptr;
				COMPUTE_SAFE(cudaMallocManaged(&temp, m_Info[1] * 2 * sizeof(T)));
				memcpy(temp, m_Data, sizeof(*m_Data) * m_Info[1]);
				COMPUTE_SAFE(cudaFree(m_Data));

				m_Data = temp;
				m_Info[1] *= 2;
			}

			m_Data[m_Info[0]++] = data;
			return m_Info[0];
		}
		__host__ __device__ T PopBack() {
			return m_Data[m_Info[0]-- - 1];
		}
		__host__ __device__ T& At(unsigned int index) {
			if (index >= m_Info[1]) {
				printf("Error: index out of range!\n");
				exit(0);
			}

			return *(m_Data + index);
		}

		// Getters
		__host__ __device__ unsigned int GetSize() {
			return m_Info[0];
		}
		__host__ __device__ unsigned int GetCapacity() {
			return m_Info[1];
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
			return Iterator<T>(m_Data + m_Info[1]);
		}
		__host__ __device__ Iterator<T> begin() {
			return Iterator<T>(m_Data);
		}
		__host__ __device__ Iterator<T> end() {
			return Iterator<T>(m_Data + m_Info[1]);
		}
	private:

	private:
		T* m_Data = nullptr;  // Array contents
		unsigned int* m_Info; // [0] = size, [1] = capacity
	};

	// TEMP
	extern "C" {
		void TestCUDA();
	}
}

#endif // !GPU_SDF_CUH