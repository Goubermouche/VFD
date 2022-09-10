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

		__host__ __device__ void Fill(T Value) {
			for (int i = 0; i < ElementCount; i++)
			{
				Grid[i] = Value;
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

		__host__ __device__ T operator()(glm::ivec3 Indices) {
			if (IsIndexInRange(Indices) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(Indices)];
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

		__host__ __device__ T Get(glm::ivec3 Indices) {
			if (IsIndexInRange(Indices) == false) {
				if (IsOutOfRangeValueSet) {
					return OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return Grid[GetFlatIndex(Indices)];
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

		__host__ __device__ void Set(int i, int j, int k, T Value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(i, j, k)] = Value;
		}

		__host__ __device__ void Set(glm::ivec3 Indices, T Value) {
			if (IsIndexInRange(Indices) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(Indices)] = Value;
		}

		__host__ void Set(std::vector<glm::ivec3>& Cells, T Value) {
			for (unsigned int i = 0; i < Cells.size(); i++) {
				Set(Cells[i], Value);
			}
		}

		__host__ __device__ void Set(int flatIndex, T Value) {
			if ((flatIndex <= 0 && flatIndex > ElementCount)) { // ! 
				printf("error: index out of range\n");
			}

			Grid[flatIndex] = Value;
		}

		__host__ __device__ void Add(int i, int j, int k, T Value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(i, j, k)] += Value;
		}

		__device__ void AtomicAdd(int i, int j, int k, T Value) {
			if (IsIndexInRange(i, j, k) == false) {
				printf("error: index out of range\n");
			}

			atomicAdd(&Grid[GetFlatIndex(i, j, k)], Value);
		}

		__host__ __device__ void Add(glm::ivec3 Indices, T Value) {
			if (IsIndexInRange(Indices) == false) {
				printf("error: index out of range\n");
			}

			Grid[GetFlatIndex(Indices)] += Value;
		}

		__host__ __device__ void Add(int flatIndex, T Value) {
			if (flatIndex <= 0 && flatIndex > ElementCount) {
				printf("error: index out of range\n");
			}

			Grid[flatIndex] += Value;
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

		__host__ __device__ T* GetPointer(glm::ivec3 Indices) {
			if (IsIndexInRange(Indices) == false) {
				if (IsOutOfRangeValueSet) {
					return &OutOfRangeValue;
				}

				printf("error: index out of range\n");
			}

			return &Grid[GetFlatIndex(Indices)];
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

		__host__ __device__ void SetOutOfRangeValue(T Value) {
			OutOfRangeValue = Value;
			IsOutOfRangeValueSet = true;
		}

		__host__ __device__ bool GetIsOutOfRangeValueSet() {
			return IsOutOfRangeValueSet;
		}

		__host__ __device__ T GetOutOfRangeValue() {
			return OutOfRangeValue;
		}

		__host__ __device__ bool IsIndexInRange(int i, int j, int k) {
			return i >= 0 && j >= 0 && k >= 0 && i < Size.x&& j < Size.y&& k < Size.z;
		}

		__host__ __device__ bool IsIndexInRange(glm::ivec3 Indices) {
			return Indices.x >= 0 && Indices.y >= 0 && Indices.z >= 0 && Indices.x < Size.x&& Indices.y < Size.y&& Indices.z < Size.z;
		}
		
		__host__ __device__ unsigned int GetFlatIndex(int i, int j, int k) {
			return (unsigned int)i + (unsigned int)Size.x *
				((unsigned int)j + (unsigned int)Size.y * (unsigned int)k);
		}

		__host__ __device__ unsigned int GetFlatIndex(glm::ivec3 Indices) {
			return (unsigned int)Indices.x + (unsigned int)Size.x *
				((unsigned int)Indices.y + (unsigned int)Size.y * (unsigned int)Indices.z);
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
		__host__ __device__ void UploadToDevice(Array3D<T>& device, const S& symbol) {
			device.ElementCount = ElementCount;
			device.Size = Size;
			device.IsOutOfRangeValueSet = IsOutOfRangeValueSet;
			device.OutOfRangeValue = OutOfRangeValue;

			cudaMalloc(&(device.Grid), ElementCount * sizeof(T));
			cudaMemcpy(device.Grid, Grid, ElementCount * sizeof(T), cudaMemcpyHostToDevice);
			COMPUTE_SAFE(cudaMemcpyToSymbol(symbol, &device, sizeof(Array3D<T>)));
		}

		__host__ __device__ void UploadToHost(Array3D<T>& host) {
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

		__host__ __device__ void DeviceFree() {
			COMPUTE_SAFE(cudaFree(Grid));
		}

		__host__ __device__ void HostFree() {
			delete[] Grid;
		}

		glm::ivec3 Size;
		bool IsOutOfRangeValueSet;
		int ElementCount;
		T OutOfRangeValue;
		T* Grid;
	};
}

#endif // !ARRAY_3D_CUH