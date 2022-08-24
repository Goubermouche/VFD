#ifndef MARKER_AND_CELL_VELOCITY_FIELD_CUH
#define MARKER_AND_CELL_VELOCITY_FIELD_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"

namespace fe {
	struct MACVelocityField {
		__device__ MACVelocityField() {}

		__device__ void Init(int i, int j, int k, float dx) {
			Size = { i, j, k };
			DX = dx;

			U.Init(Size.x + 1, Size.y, Size.z, 1.0f);
			V.Init(Size.x, Size.y + 1, Size.z, 2.0f);
			W.Init(Size.x, Size.y, Size.z + 1, 3.0f);

			U.SetOutOfRangeValue(0.0f);
			V.SetOutOfRangeValue(0.0f);
			W.SetOutOfRangeValue(0.0f);

			LOG("velocity grids initialized", "FLIP][MAC", ConsoleColor::Cyan);
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