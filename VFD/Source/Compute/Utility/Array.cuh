#ifndef ARRAY_CUH
#define ARRAY_CUH

#include "pch.h"
#include "host_defines.h"
#include "cuda_runtime.h"
#include "Debug/SystemInfo.h"

namespace vfd {
	template<class T>
	struct Iterator {
		__host__ __device__ Iterator() {}

		__host__ __device__ Iterator(T* ptr)
			: m_Ptr(ptr)
		{}

		__host__ __device__ bool operator==(const Iterator& rhs) const {
			return m_Ptr == rhs.m_Ptr;
		}

		__host__ __device__ bool operator!=(const Iterator& rhs) const {
			return !(*this == rhs);
		}

		__host__ __device__ T operator*() const {
			return *m_Ptr;
		}

		__host__ __device__ Iterator& operator++() {
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

	/// <summary>
	/// Host/device dynamic array.
	/// </summary>
	/// <typeparam name="T">Data type</typeparam>
	template<class T>
	struct Array {
#pragma region Constructors
		__host__ Array() {
			ASSERT(SystemInfo::CUDADeviceMeetsRequirements(), "Cuda device does not meet capability requirements!");
			COMPUTE_SAFE(cudaMallocManaged(&m_Data, sizeof(T)));
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 2 * sizeof(unsigned int)));

			m_Info[0] = 0;
			m_Info[1] = 1;
		}

		__host__ Array(const std::vector<T>& vec) {
			ASSERT(SystemInfo::CUDADeviceMeetsRequirements(), "Cuda device does not meet capability requirements!");
			COMPUTE_SAFE(cudaMallocManaged(&m_Data, vec.capacity() * sizeof(T)));
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 2 * sizeof(unsigned int)));
			memcpy(m_Data, vec.data(), vec.capacity() * sizeof(T));

			m_Info[0] = vec.size();
			m_Info[1] = vec.capacity();
		}

		/// <summary>
		/// Creates a new array with the specified capacity.
		/// </summary>
		/// <param name="capacity">Target capacity of the array</param>
		__host__ Array(unsigned int capacity) {
			ASSERT(SystemInfo::CUDADeviceMeetsRequirements(), "Cuda device does not meet capability requirements!");
			ASSERT(capacity > 0, "Array size must be greater than 0!");
			COMPUTE_SAFE(cudaMallocManaged(&m_Data, capacity * sizeof(T)));
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 2 * sizeof(unsigned int)));

			m_Info[0] = 0;
			m_Info[1] = capacity;
		}

		/// <summary>
		/// Creates a new array with the specified capacity and fills it with the specified value.
		/// </summary>
		/// <param name="capacity">Target capacity of the array</param>
		/// <param name="value">Fill value</param>
		/// <returns></returns>
		__host__ Array(unsigned int capacity, const T& value) {
			ASSERT(SystemInfo::CUDADeviceMeetsRequirements(), "Cuda device does not meet capability requirements!");
			ASSERT(capacity > 0, "Array size must be greater than 0!");
			COMPUTE_SAFE(cudaMallocManaged(&m_Data, capacity * sizeof(T)));
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 2 * sizeof(unsigned int)));

			m_Info[0] = 0;
			m_Info[1] = capacity;
			Fill(value);
		}

#pragma endregion

#pragma region Public member functions
		/// <summary>
		/// Frees the allocated unified memory for this array.
		/// </summary>
		__host__ void Free() {
			COMPUTE_SAFE(cudaFree(m_Data));
			COMPUTE_SAFE(cudaFree(m_Info));
		}

		/// <summary>
		/// Fills the entire array with the specified value.
		/// </summary>
		/// <param name="value">Fill value</param>
		__host__ __device__ void Fill(const T& value) {
			m_Info[0] = m_Info[1];

			for (unsigned int i = 0; i < m_Info[1]; i++)
			{
				m_Data[i] = value;
			}
		}

		/// <summary>
		/// Checks if the array size is equal to 0, note that if the array contains 
		/// accessor defined elements this function will become inaccurate.
		/// </summary>
		__host__ __device__ bool IsEmpty() {
			return m_Info[0] == 0;
		}

		/// <summary>
		/// Fills the array with the specified value in the specified range.
		/// </summary>
		/// <param name="value">Fill value</param>
		/// <param name="start">Start index of the fill operation</param>
		/// <param name="end">End index of the fill operation</param>
		__host__ __device__ void Fill(const T& value, unsigned int start, unsigned int end) {
			if (end > m_Info[1]) {
				printf("Error: end must be contained within the container!\n");
				exit(0);
			}

			m_Info[0] = end;
			for (unsigned int i = start; i < end; i++)
			{
				m_Data[i] = value;
			}
		}

		/// <summary>
		/// Removes all elements from the array.
		/// </summary>
		__host__ __device__ void Clear() {
			for (unsigned int i = 0; i < m_Info[1]; i++)
			{
				m_Data[i] = T();
			}

			m_Info[0] = 0;
		}

		/// <summary>
		/// Pops the last array element.
		/// </summary>
		/// <returns>Popped element</returns>
		__host__ __device__ T RemoveLastElement() {
			if (m_Info[0] == 0) {
				printf("Error: index out of range!\n");
				exit(0);
			}

			return m_Data[m_Info[0]-- - 1];
		}

		/// <summary>
		/// Deletes an element at the specified index.
		/// </summary>
		/// <param name="index">Index to delete the element at</param>
		/// <returns>Deleted element</returns>
		__host__ __device__ T DeleteElement(const unsigned int index) {
			// TODO: Check for atomicity
			if (index > m_Info[0]) {
				printf("Error: index out of range!\n");
				exit(0);
			}

			T tmp = m_Data[index];

			for (unsigned int i = index; i < m_Info[0] - 1; ++i) {
				m_Data[i] = m_Data[i + 1];
			}

			m_Info[0]--;
			return tmp;
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ T& At(unsigned int index) {
			if (index >= m_Info[1]) {
				printf("Error: index out of range!\n");
				exit(0);
			}

			return *(m_Data + index);
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ const T& At(unsigned int index) const {
			if (index >= m_Info[1]) {
				printf("Error: index out of range!\n");
				exit(0);
			}

			return *(m_Data + index);
		}

		/// <summary>
		/// Expands the memory buffer that is allocated for the array.
		/// </summary>
		/// <param name="capacity">New array capacity</param>
		__host__ void Reserve(unsigned int capacity) {
			if (capacity <= m_Info[1]) {
				return;
			}

			T* temp = nullptr;
			COMPUTE_SAFE(cudaMallocManaged(&temp, capacity * sizeof(T)));
			memcpy(temp, m_Data, sizeof(*m_Data) * m_Info[1]);
			COMPUTE_SAFE(cudaFree(m_Data));

			m_Data = temp;
			m_Info[1] = capacity;
		}

		/// <summary>
		/// Resizes the entire array, if the new size is greater than the current capacity 
		/// the array expands. Elements located outside the old bounds will be set to the 
		/// fill value, if one is provided.
		/// </summary>
		/// <param name="size">New array size</param>
		/// <param name="value">Fill value</param>
		__host__ void Resize(unsigned int size, const T& value = T()) {
			if (size > m_Info[1]) {
				Reserve(size);
				Fill(value, m_Info[0], size);
			}
			else {
				Fill(value, m_Info[0], size);
			}

			m_Info[0] = size;
		}

		/// <summary>
		/// Adds a new element to the end of the array. 
		/// </summary>
		/// <param name="data">Element to add</param>
		/// <returns>New array size</returns>
		__host__ unsigned int AddElement(const T& data) {
			if (m_Info[0] == m_Info[1]) {
				Reserve(m_Info[1] * 2);
			}

			m_Data[m_Info[0]++] = data;
			return m_Info[0];
		}
#pragma endregion

#pragma region Getters
		__host__ __device__ const unsigned int GetSize() const {
			return m_Info[0];
		}

		__host__ __device__ const unsigned int GetCapacity() const {
			return m_Info[1];
		}

		__host__ __device__ T* GetData() {
			return m_Data;
		}
#pragma endregion

#pragma region Overloads
		__host__ __device__ T& operator[](unsigned int index) {
			return At(index);
		}

		// NOTE: iterators have to be lowercase in order for them to get picked up by 
		//       the compiler.
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

#pragma endregion
	private:
	private:
		T* m_Data = nullptr;  // Array contents
		unsigned int* m_Info; // { size, capacity }
	};
}

#endif // !ARRAY_CUH