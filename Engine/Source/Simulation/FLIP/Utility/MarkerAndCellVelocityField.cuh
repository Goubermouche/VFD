#ifndef MARKER_AND_CELL_VELOCITY_FIELD_CUH
#define MARKER_AND_CELL_VELOCITY_FIELD_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"

namespace fe {
	struct ValidVelocityComponent {
		__device__ ValidVelocityComponent() {}

		__device__ void Init(int i, int j, int k) {
			ValidU.Init(i + 1, j, k, false);
			ValidU.Init(i, j + 1, k, false);
			ValidU.Init(i, j, k + 1, false);
		}

		__device__ void Reset() {
			ValidU.Fill(false);
			ValidV.Fill(false);
			ValidW.Fill(false);
		}

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

		__device__ void Free() {
			COMPUTE_SAFE(cudaFree(U.Grid));
			COMPUTE_SAFE(cudaFree(V.Grid));
			COMPUTE_SAFE(cudaFree(W.Grid));
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