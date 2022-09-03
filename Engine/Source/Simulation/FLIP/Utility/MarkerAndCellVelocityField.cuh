#ifndef MARKER_AND_CELL_VELOCITY_FIELD_CUH
#define MARKER_AND_CELL_VELOCITY_FIELD_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/Grid3D.cuh"

namespace fe {
	struct ValidVelocityComponent {
		__device__ ValidVelocityComponent() {}

		__device__ void Init(int i, int j, int k) {
			ValidU.Init(i + 1, j, k, false);
			ValidV.Init(i, j + 1, k, false);
			ValidW.Init(i, j, k + 1, false);
		}

		__device__ void Reset() {
			ValidU.Fill(false);
			ValidV.Fill(false);
			ValidW.Fill(false);
		}

		// TODO: add free methods

		Array3D<bool> ValidU;
		Array3D<bool> ValidV;
		Array3D<bool> ValidW;
	};

	struct MACVelocityField {
		__device__ MACVelocityField() {}

		__device__ void Init(int i, int j, int k, float dx) {
			Size = { i, j, k };
			DX = dx;

			U.Init(Size.x + 1, Size.y, Size.z, 0.0f);
			V.Init(Size.x, Size.y + 1, Size.z, 0.0f);
			W.Init(Size.x, Size.y, Size.z + 1, 0.0f);

			U.SetOutOfRangeValue(0.0f);
			V.SetOutOfRangeValue(0.0f);
			W.SetOutOfRangeValue(0.0f);

			LOG("velocity grids initialized", "FLIP", ConsoleColor::Cyan);
		}

		__host__ MACVelocityField UploadToDevice() {
			MACVelocityField device = *this;

			// Perform a deep copy on the individual arrays
			device.U.UploadToDevice(device.U);
			device.V.UploadToDevice(device.V);
			device.W.UploadToDevice(device.W);

			return device;
		}

		__device__ __host__ bool IsIndexInRangeU(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x + 1, Size.y, Size.z);
		}
		__device__ __host__ bool IsIndexInRangeV(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x, Size.y + 1, Size.z);
		}
		__device__ __host__ bool IsIndexInRangeW(int i, int j, int k) {
			return IsGridIndexInRange({ i, j, k }, Size.x, Size.y, Size.z + 1);
		}

		__device__ __host__ void SetU(int i, int j, int k, float val) {
			if (!IsIndexInRangeU(i, j, k)) {
				return;
			}

			U.Set(i, j, k, (float)val);
		}

		__device__ __host__ void SetV(int i, int j, int k, float val) {
			if (!IsIndexInRangeV(i, j, k)) {
				return;
			}

			V.Set(i, j, k, (float)val);
		}

		__device__ __host__ void SetW(int i, int j, int k, float val) {
			if (!IsIndexInRangeW(i, j, k)) {
				return;
			}

			W.Set(i, j, k, (float)val);
		}

		__device__ __host__ void ClearU() {
			U.Fill(0.0);
		}

		__device__ __host__ void ClearV() {
			V.Fill(0.0);
		}

		__device__ __host__ void ClearW() {
			W.Fill(0.0);
		}

		__device__ __host__ void Clear() {
			ClearU();
			ClearV();
			ClearW();
		}

		__host__ void DeviceFree() {
			U.DeviceFree();
			V.DeviceFree();
			W.DeviceFree();
		}

		__host__ void HostFree() {
			U.HostFree();
			V.HostFree();
			W.HostFree();
		}

		Array3D<float> U;
		Array3D<float> V;
		Array3D<float> W;

		float DefaultOutOfRangeValue;
		float DX;

		int ExtrapolationLayerCount;

		glm::ivec3 Size;
	};
}

#endif // !MARKER_AND_CELL_VELOCITY_FIELD_CUH