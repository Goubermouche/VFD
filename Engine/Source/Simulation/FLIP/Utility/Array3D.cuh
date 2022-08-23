#ifndef ARRAY_3D_CUH
#define ARRAY_3D_CUH

#include "pch.h"

namespace fe {
	template<class T>
	struct Array3D {
		__device__ Array3D() {
		}

		__device__ Array3D(int i, int j, int k)
			: m_Size({ i, j, k }) {
			InitializeGrid();
		}

		__device__ Array3D(int i, int j, int k, T fillValue)
			: m_Size({ i, j, k }) {
			InitializeGrid();
			Fill(fillValue);
		}

		__device__ Array3D(const Array3D& obj) {
			m_Size = obj.m_Size;
			m_ElementCount = obj.m_ElementCount;

			InitializeGrid();

			T val;
			for (int k = 0; k < m_Size.z; k++)
			{
				for (int j = 0; j < m_Size.y; j++)
				{
					for (int i = 0; i < m_Size.x; i++)
					{
						val = obj.m_Grid[GetFlatIndex(i, j, k)];
						Set(i, j, k, val);
					}
				}
			}

			if (obj.m_IsOutOfRangeValueSet) {
				m_OutOfRangeValue = obj.m_OutOfRangeValue;
				m_IsOutOfRangeValueSet = true;
			}
		}

		__device__ Array3D operator=(const Array3D& rhs) {
			delete[] m_Grid;

			m_Size = rhs.m_Size;
			m_ElementCount = rhs.m_ElementCount;

			InitializeGrid();

			T val;
			for (int k = 0; k < m_Size.z; k++)
			{
				for (int j = 0; j < m_Size.y; j++)
				{
					for (int i = 0; i < m_Size.x; i++)
					{
						val = rhs.m_Grid[GetFlatIndex(i, j, k)];
						Set(i, j, k, val);
					}
				}
			}

			if (rhs.m_IsOutOfRangeValueSet) {
				m_OutOfRangeValue = rhs.m_OutOfRangeValue;
				m_IsOutOfRangeValueSet = true;
			}

			return *this;
		}

		__device__ void SetDefault() {
			m_Grid = nullptr;
			m_Size = { 0, 0, 0 };
			m_ElementCount = 0;
		}

		__device__ void Fill(T value) {
			for (int i = 0; i < m_Size.x * m_Size.y * m_Size.z; i++)
			{
				m_Grid[i] = value;
			}
		}

		__device__ T operator()(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (m_IsOutOfRangeValueSet) {
					return m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return m_Grid[GetFlatIndex(i, j, k)];
		}

		__device__ T operator()(glm::ivec3 g) {
			if (IsIndexInRange(g) == false) {
				if (m_IsOutOfRangeValueSet) {
					return m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return m_Grid[GetFlatIndex(g)];
		}

		__device__ T operator()(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > m_ElementCount) { // ! 
				if (m_IsOutOfRangeValueSet) {
					return m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return m_Grid[flatIndex];
		}

		__device__ T Get(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (m_IsOutOfRangeValueSet) {
					return m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return m_Grid[GetFlatIndex(i, j, k)];
		}

		__device__ T Get(glm::ivec3 g) {
			if (IsIndexInRange(g) == false) {
				if (m_IsOutOfRangeValueSet) {
					return m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return m_Grid[GetFlatIndex(g)];
		}

		__device__ T Get(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > m_ElementCount) { // !
				if (m_IsOutOfRangeValueSet) {
					return m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return m_Grid[flatIndex];
		}

		__device__ void Set(int i, int j, int k, T value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			m_Grid[GetFlatIndex(i, j, k)] = value;
		}

		__device__ void Set(glm::ivec3 g, T value) {
			if (IsIndexInRange(g) == false) {
				printf("error: index out of range\n");
			}

			m_Grid[GetFlatIndex(g)] = value;
		}

		__device__ void Set(int flatIndex, T value) {
			if ((flatIndex <= 0 && flatIndex > m_ElementCount)) { // ! 
				printf("error: index out of range\n");
			}

			m_Grid[flatIndex] = value;
		}

		__device__ void Add(int i, int j, int k, T value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			m_Grid[GetFlatIndex(i, j, k)] += value;
		}

		__device__ void Add(glm::ivec3 g, T value) {
			if (IsIndexInRange(g) == false) {
				printf("error: index out of range\n");
			}

			m_Grid[GetFlatIndex(g)] += value;
		}

		__device__ void Add(int flatIndex, T value) {
			if (flatIndex <= 0 && flatIndex > m_ElementCount) {
				printf("error: index out of range\n");
			}

			m_Grid[flatIndex] += value;
		}

		__device__ T* GetPointer(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (m_IsOutOfRangeValueSet) {
					return &m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &m_Grid[GetFlatIndex(i, j, k)];
		}

		__device__ T* GetPointer(glm::ivec3 g) {
			if (IsIndexInRange(g) == false) {
				if (m_IsOutOfRangeValueSet) {
					return &m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &m_Grid[GetFlatIndex(g)];
		}

		__device__ T* GetPointer(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > m_ElementCount) {
				if (m_IsOutOfRangeValueSet) {
					return &m_OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &m_Grid[flatIndex];
		}

		__device__ T* GetRawArray() {
			return m_Grid;
		}

		__device__ int GetElementCount() {
			return m_ElementCount;
		}

		__device__ void SetOutOfRangeValue() {
			m_IsOutOfRangeValueSet = false;
		}

		__device__ void SetOutOfRangeValue(T value) {
			m_OutOfRangeValue = value;
			m_IsOutOfRangeValueSet = true;
		}

		__device__ bool IsOutOfRangeValueSet() {
			return m_IsOutOfRangeValueSet;
		}

		__device__ T GetOutOfRangeValue() {
			return m_OutOfRangeValue;
		}

		inline __device__ bool IsIndexInRange(int i, int j, int k) {
			return i >= 0 && j >= 0 && k >= 0 && i < m_Size.x&& j < m_Size.y&& k < m_Size.z;
		}

		inline __device__ bool IsIndexInRange(glm::ivec3 g) {
			return g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < m_Size.x&& g.y < m_Size.y&& g.z < m_Size.z;
		}
	private:
		__device__ void InitializeGrid() {
			if (m_Size.x < 0 || m_Size.y < 0 || m_Size.z < 0) {
				printf("%d\n", m_Size.x);
				printf("%d\n", m_Size.y);
				printf("%d\n", m_Size.z);
				printf("error: dimensions cannot be negative.\n");
			}

			m_Grid = new T[m_Size.x * m_Size.y * m_Size.z];
		}

		inline __device__ unsigned int GetFlatIndex(int i, int j, int k) {
			return (unsigned int)i + (unsigned int)m_Size.x *
				((unsigned int)j + (unsigned int)m_Size.y * (unsigned int)k);
		}

		inline __device__ unsigned int GetFlatIndex(glm::ivec3 g) {
			return (unsigned int)g.x + (unsigned int)m_Size.x *
				((unsigned int)g.y + (unsigned int)m_Size.y * (unsigned int)g.z);
		}

		glm::ivec3 m_Size;
		int m_ElementCount;
		bool m_IsOutOfRangeValueSet;
		T m_OutOfRangeValue;
		T* m_Grid;
	};
}

#endif // !ARRAY_3D_CUH