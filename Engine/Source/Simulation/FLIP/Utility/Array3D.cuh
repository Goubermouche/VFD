#ifndef ARRAY_3D_CUH
#define ARRAY_3D_CUH

#include "pch.h"

namespace fe {
	template<class T>
	struct Array3D {
		__device__ Array3D() {}

		__device__ void Init(int i, int j, int k) {
			Size = { i, j, k };
			ElementCount = i * j * k;
			Grid = new T[ElementCount];
		}

		__device__ void Init(int i, int j, int k, T fillValue) {
			Size = { i, j, k };
			ElementCount = i * j * k;
			Grid = new T[ElementCount];
			Fill(fillValue);
		}

		__device__ void Fill(T value) {
			for (int i = 0; i < ElementCount; i++)
			{
				Grid[i] = value;
			}
		}

		__device__ void Clear() {
			delete[] Grid;
		}

		__device__ T operator()(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(i, j, k)];
		}

		__device__ T operator()(glm::ivec3 g) {
			if (IsIndexInRange(g) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(g)];
		}

		__device__ T operator()(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > ElementCount) { // ! 
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[flatIndex];
		}

		__device__ T Get(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(i, j, k)];
		}

		__device__ T Get(glm::ivec3 g) {
			if (IsIndexInRange(g) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(g)];
		}

		__device__ T Get(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > ElementCount) { // !
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[flatIndex];
		}

		__device__ void Set(int i, int j, int k, T value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(i, j, k)] = value;
		}

		__device__ void Set(glm::ivec3 g, T value) {
			if (IsIndexInRange(g) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(g)] = value;
		}

		__device__ void Set(int flatIndex, T value) {
			if ((flatIndex <= 0 && flatIndex > ElementCount)) { // ! 
				printf("error: index out of range\n");
			}

			Grid[flatIndex] = value;
		}

		__device__ void Add(int i, int j, int k, T value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(i, j, k)] += value;
		}

		__device__ void Add(glm::ivec3 g, T value) {
			if (IsIndexInRange(g) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(g)] += value;
		}

		__device__ void Add(int flatIndex, T value) {
			if (flatIndex <= 0 && flatIndex > ElementCount) {
				printf("error: index out of range\n");
			}

			Grid[flatIndex] += value;
		}

		__device__ T* GetPointer(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (IsOutOfRangeValueSet) {
					return &OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &Grid[GetFlatIndex(i, j, k)];
		}

		__device__ T* GetPointer(glm::ivec3 g) {
			if (IsIndexInRange(g) == false) {
				if (IsOutOfRangeValueSet) {
					return &OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &Grid[GetFlatIndex(g)];
		}

		__device__ T* GetPointer(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > ElementCount) {
				if (IsOutOfRangeValueSet) {
					return &OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &Grid[flatIndex];
		}

		__device__ T* GetRawArray() {
			return Grid;
		}

		__device__ int GetElementCount() {
			return ElementCount;
		}

		__device__ unsigned int GetSize() {
			return sizeof(T) * ElementCount;
		}

		__device__ void SetOutOfRangeValue() {
			IsOutOfRangeValueSet = false;
		}

		__device__ void SetOutOfRangeValue(T value) {
			OutOfRangeValue = value;
			IsOutOfRangeValueSet = true;
		}

		__device__ bool GetIsOutOfRangeValueSet() {
			return IsOutOfRangeValueSet;
		}

		__device__ T GetOutOfRangeValue() {
			return OutOfRangeValue;
		}

		inline __device__ bool IsIndexInRange(int i, int j, int k) {
			return i >= 0 && j >= 0 && k >= 0 && i < Size.x&& j < Size.y&& k < Size.z;
		}

		inline __device__ bool IsIndexInRange(glm::ivec3 g) {
			return g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < Size.x&& g.y < Size.y&& g.z < Size.z;
		}
		
		inline __device__ unsigned int GetFlatIndex(int i, int j, int k) {
			return (unsigned int)i + (unsigned int)Size.x *
				((unsigned int)j + (unsigned int)Size.y * (unsigned int)k);
		}

		inline __device__ unsigned int GetFlatIndex(glm::ivec3 g) {
			return (unsigned int)g.x + (unsigned int)Size.x *
				((unsigned int)g.y + (unsigned int)Size.y * (unsigned int)g.z);
		}

		glm::ivec3 Size;
		T* Grid;
		int ElementCount;
		bool IsOutOfRangeValueSet;
		T OutOfRangeValue;
	};

	struct MAC {
		__device__ MAC() {}

		Array3D<float> Arr1;
		Array3D<float> Arr2;
		Array3D<float> Arr3;
	};
}

#endif // !ARRAY_3D_CUH