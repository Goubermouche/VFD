#ifndef ARRAY_3D_CUH
#define ARRAY_3D_CUH

#include "Compute/Utility/Array.cuh"

namespace vfd {
	/// <summary>
	/// Host/device dynamic 3D array.
	/// </summary>
	/// <typeparam name="T">Data type</typeparam>
	template<class T>
	struct Array3D {
#pragma region Constructors
		__host__ Array3D() {
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 3 * sizeof(unsigned int)));
			m_Array = Array<T>(1);

			m_Info[0] = 1;
			m_Info[1] = 1;
			m_Info[2] = 1;
		}

		/// <summary>
		/// Creates a new 3D array with the specified capacity.
		/// </summary>
		/// <param name="capacity">Target capacity of the array</param>
		__host__ Array3D(const glm::uvec3& capacity) {
			ASSERT(capacity.x > 0 && capacity.y > 0 && capacity.z > 0, "size must be greater than 0!");
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 3 * sizeof(unsigned int)));
			m_Array = Array<T>(capacity.x * capacity.y * capacity.z);

			m_Info[0] = capacity.x;
			m_Info[1] = capacity.y;
			m_Info[2] = capacity.z;
		}

		/// <summary>
		/// Creates a new 3D array with the specified capacity and fills it with the specified value.
		/// </summary>
		/// <param name="capacity">Target capacity of the array</param>
		/// <param name="value">Fill value</param>
		/// <returns></returns>
		__host__ Array3D(const glm::uvec3& capacity, const T& value) {
			ASSERT(capacity.x > 0 && capacity.y > 0 && capacity.z > 0, "size must be greater than 0!");
			COMPUTE_SAFE(cudaMallocManaged(&m_Info, 3 * sizeof(unsigned int)));
			m_Array = Array<T>(capacity.x * capacity.y * capacity.z, value);

			m_Info[0] = capacity.x;
			m_Info[1] = capacity.y;
			m_Info[2] = capacity.z;
		}
#pragma endregion

#pragma region Public member functions
		/// <summary>
		/// Frees the allocated unified memory for this 3D array.
		/// </summary>
		__host__ void Free() {
			COMPUTE_SAFE(cudaFree(m_Info));
			m_Array.Free();
		}

		/// <summary>
		/// Resizes the entire 3D array, if the new size is greater than the current capacity 
		/// the array expands. Elements located outside the old bounds will be set to the fill
		/// value, if one is provided.
		/// </summary>
		/// <param name="size">New array size</param>
		/// <param name="value">Fill value</param>
		__host__ void Resize(const glm::uvec3& capacity, const T& value = T()) {
			m_Array.Resize(capacity.x * capacity.y * capacity.z, value);

			m_Info[0] = capacity.x;
			m_Info[1] = capacity.y;
			m_Info[2] = capacity.z;
		}

		/// <summary>
		/// Fills the entire 3D array with the specified value.
		/// </summary>
		/// <param name="value">Fill value</param>
		__host__ __device__ void Fill(const T& value) {
			m_Array.Fill(value);
		}

		/// <summary>
		/// Checks if the specified index is within the array's bounds. 
		/// </summary>
		__host__ __device__ const bool IsIndexInRange(const glm::uvec3& index) const {
			return index.x < m_Info[0] && index.y < m_Info[1] && index.z < m_Info[2];
		}

		/// <summary>
		/// Checks if the specified index is within the array's bounds. 
		/// </summary>
		__host__ __device__ const bool IsIndexInRange(unsigned int x, unsigned int y, unsigned int z) const {
			return x < m_Info[0] && y < m_Info[1] && z < m_Info[2];
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ const T& At(const glm::uvec3& index) const {
			return m_Array.At(index.x + m_Info[0] * (index.y + m_Info[1] * index.z));
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ T& At(const glm::uvec3& index) {
			return m_Array.At(index.x + m_Info[0] * (index.y + m_Info[1] * index.z));
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ const T& At(unsigned int x, unsigned int y, unsigned int z) const {
			return m_Array.At(x + m_Info[0] * (y + m_Info[1] * z));
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ T& At(unsigned int x, unsigned int y, unsigned int z) {
			return m_Array.At(x + m_Info[0] * (y + m_Info[1] * z));
		}
#pragma endregion

#pragma region Getters
		__host__ __device__ unsigned int GetSize() {
			return m_Array.GetSize();
		}

		__host__ __device__ unsigned int GetSizeX() {
			return m_Info[0];
		}

		__host__ __device__ unsigned int GetSizeY() {
			return m_Info[1];
		}

		__host__ __device__ unsigned int GetSizeZ() {
			return m_Info[2];
		}

		__host__ __device__ unsigned int GetCapacity() {
			return m_Array.GetCapacity();
		}

		__host__ __device__ T* GetData() {
			return m_Array.GetData();
		}
#pragma endregion

#pragma region Overloads
		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ const T& operator()(const glm::uvec3& index) const
		{
			return At(index);
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ T& operator()(const glm::uvec3& index)
		{
			return At(index);
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ const T& operator()(unsigned int x, unsigned int y, unsigned int z) const
		{
			return At(x, y, z);
		}

		/// <summary>
		/// Returns the element located at the specified index.
		/// </summary>
		/// <param name="index">Index with the target element</param>
		/// <returns>Element located at the specified index</returns>
		__host__ __device__ T& operator()(unsigned int x, unsigned int y, unsigned int z)
		{
			return At(x, y, z);
		}

		T operator [] (int i) const {
			return m_Array[i];
		}

		T& operator [] (int i) {
			return m_Array[i];
		}

		// NOTE: iterators have to be lowercase in order for them to get picked up by 
		//       the compiler.
		__host__ __device__ Iterator<T> begin() const {
			return Iterator<T>(m_Array.begin());
		}

		__host__ __device__ Iterator<T> end() const {
			return Iterator<T>(m_Array.end());
		}

		__host__ __device__ Iterator<T> begin() {
			return Iterator<T>(m_Array.begin());
		}

		__host__ __device__ Iterator<T> end() {
			return Iterator<T>(m_Array.end());
		}
#pragma endregion
	private:
		Array<T> m_Array;     // Underlying base array
		unsigned int* m_Info; // { sizeX, sizeY, sizeZ }
	};
}

#endif // !ARRAY_3D_CUH