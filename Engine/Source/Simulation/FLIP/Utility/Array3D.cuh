#ifndef ARRAY_3D_CUH
#define ARRAY_3D_CUH

#include "pch.h"

namespace fe {
	template<class T>
	struct Array3D {
		__host__ __device__ Array3D() {}

		__host__ __device__ void Init(int i, int j, int k) {
			Size = { i, j, k };
			ElementCount = i * j * k;
			Grid = new T[ElementCount];
		}

		__host__ __device__ void Init(int i, int j, int k, T fillValue) {
			Size = { i, j, k };
			ElementCount = i * j * k;
			Grid = new T[ElementCount];
			Fill(fillValue);
		}

		__host__ __device__ void Fill(T value) {
			for (int i = 0; i < ElementCount; i++)
			{
				Grid[i] = value;
			}
		}

		__host__ __device__ void Clear() {
			delete[] Grid;
		}

		__host__ __device__ T operator()(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(i, j, k)];
		}

		__host__ __device__ T operator()(glm::ivec3 index) {
			if (IsIndexInRange(index) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(index)];
		}

		__host__ __device__ T operator()(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > ElementCount) { // ! 
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[flatIndex];
		}

		__host__ __device__ T Get(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(i, j, k)];
		}

		__host__ __device__ T Get(glm::ivec3 index) {
			if (IsIndexInRange(index) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(index)];
		}

		__host__ __device__ T Get(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > ElementCount) { // !
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[flatIndex];
		}

		__host__ __device__ void Set(int i, int j, int k, T value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(i, j, k)] = value;
		}

		__host__ __device__ void Set(glm::ivec3 index, T value) {
			if (IsIndexInRange(index) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(index)] = value;
		}

		__host__ __device__ void Set(int flatIndex, T value) {
			if ((flatIndex <= 0 && flatIndex > ElementCount)) { // ! 
				printf("error: index out of range\n");
			}

			Grid[flatIndex] = value;
		}

		__host__ __device__ void Add(int i, int j, int k, T value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(i, j, k)] += value;
		}

		__device__ void AtomicAdd(int i, int j, int k, T value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			atomicAdd(&Grid[GetFlatIndex(i, j, k)], value);
		}

		__host__ __device__ void Add(glm::ivec3 index, T value) {
			if (IsIndexInRange(index) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(index)] += value;
		}

		__host__ __device__ void Add(int flatIndex, T value) {
			if (flatIndex <= 0 && flatIndex > ElementCount) {
				printf("error: index out of range\n");
			}

			Grid[flatIndex] += value;
		}

		__host__ __device__ T* GetPointer(int i, int j, int k) {
			if (IsIndexInRange(i, j, k) == false) {
				if (IsOutOfRangeValueSet) {
					return &OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &Grid[GetFlatIndex(i, j, k)];
		}

		__host__ __device__ T* GetPointer(glm::ivec3 index) {
			if (IsIndexInRange(index) == false) {
				if (IsOutOfRangeValueSet) {
					return &OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &Grid[GetFlatIndex(index)];
		}

		__host__ __device__ T* GetPointer(int flatIndex) {
			if (flatIndex <= 0 && flatIndex > ElementCount) {
				if (IsOutOfRangeValueSet) {
					return &OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &Grid[flatIndex];
		}

		__host__ __device__ T* GetRawArray() {
			return Grid;
		}

		__host__ __device__ int GetElementCount() {
			return ElementCount;
		}

		__host__ __device__ unsigned int GetSize() {
			return sizeof(T) * ElementCount;
		}

		__host__ __device__ void SetOutOfRangeValue() {
			IsOutOfRangeValueSet = false;
		}

		__host__ __device__ void SetOutOfRangeValue(T value) {
			OutOfRangeValue = value;
			IsOutOfRangeValueSet = true;
		}

		__host__ __device__ bool GetIsOutOfRangeValueSet() {
			return IsOutOfRangeValueSet;
		}

		__host__ __device__ T GetOutOfRangeValue() {
			return OutOfRangeValue;
		}

		inline __host__ __device__ bool IsIndexInRange(int i, int j, int k) {
			return i >= 0 && j >= 0 && k >= 0 && i < Size.x&& j < Size.y&& k < Size.z;
		}

		inline __host__ __device__ bool IsIndexInRange(glm::ivec3 index) {
			return index.x >= 0 && index.y >= 0 && index.z >= 0 && index.x < Size.x&& index.y < Size.y&& index.z < Size.z;
		}
		
		inline __host__ __device__ unsigned int GetFlatIndex(int i, int j, int k) {
			return (unsigned int)i + (unsigned int)Size.x *
				((unsigned int)j + (unsigned int)Size.y * (unsigned int)k);
		}

		inline __host__ __device__ unsigned int GetFlatIndex(glm::ivec3 index) {
			return (unsigned int)index.x + (unsigned int)Size.x *
				((unsigned int)index.y + (unsigned int)Size.y * (unsigned int)index.z);
		}

		__host__ void UploadToDevice(Array3D<T>& device) {
			device.ElementCount = ElementCount;
			device.Size = Size;
			device.IsOutOfRangeValueSet = IsOutOfRangeValueSet;
			device.OutOfRangeValue = OutOfRangeValue;

			cudaMalloc(&(device.Grid), ElementCount * sizeof(T));
			cudaMemcpy(device.Grid, Grid, ElementCount * sizeof(T), cudaMemcpyHostToDevice);
		}

		template <class S> 
		__host__ void UploadToDevice(Array3D<T>& device, const S& symbol) {
			device.ElementCount = ElementCount;
			device.Size = Size;
			device.IsOutOfRangeValueSet = IsOutOfRangeValueSet;
			device.OutOfRangeValue = OutOfRangeValue;

			cudaMalloc(&(device.Grid), ElementCount * sizeof(T));
			cudaMemcpy(device.Grid, Grid, ElementCount * sizeof(T), cudaMemcpyHostToDevice);
			COMPUTE_SAFE(cudaMemcpyToSymbol(symbol, &device, sizeof(Array3D<T>)));
		}

		__host__ void UploadToHost(Array3D<T>& host) {
			// Delete the array if it isn't empty 
			// We have to use != 1 and not != nullptr or != 0 for some reason (?)
			if (host.ElementCount != 1) {
				delete[] host.Grid;
			}

			host.ElementCount = ElementCount;
			host.Size = Size;
			host.IsOutOfRangeValueSet = IsOutOfRangeValueSet;
			host.OutOfRangeValue = OutOfRangeValue;

			host.Grid = new T[ElementCount];
			cudaMemcpy(host.Grid, Grid, host.ElementCount * sizeof(T), cudaMemcpyDeviceToHost);
		}

		__host__ __device__ void Free() {
			COMPUTE_SAFE(cudaFree(Grid));
		}

		glm::ivec3 Size;

		bool IsOutOfRangeValueSet;

		int ElementCount;

		T OutOfRangeValue;
		T* Grid;
	};
}

#endif // !ARRAY_3D_CUH