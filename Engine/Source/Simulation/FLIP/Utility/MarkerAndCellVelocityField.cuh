#ifndef MARKER_AND_CELL_VELOCITY_FIELD_CUH
#define MARKER_AND_CELL_VELOCITY_FIELD_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"

namespace fe {
	struct MACVelocityField {
		__device__ MACVelocityField() {}
		__device__ MACVelocityField(int i, int j, int k, float dx)
			: m_Size({ i, j, k }), m_DX(dx) {
			InitializeVelocityGrids();
		}

		__device__ void InitializeVelocityGrids() {
			m_U = Array3D<float>(m_Size.x + 1, m_Size.y, m_Size.z, 0.0f);
			m_V = Array3D<float>(m_Size.x, m_Size.y + 1, m_Size.z, 0.0f);
			m_W = Array3D<float>(m_Size.x, m_Size.y, m_Size.z + 1, 0.0f);

			m_U.SetOutOfRangeValue(0.0f);
			m_V.SetOutOfRangeValue(0.0f);
			m_W.SetOutOfRangeValue(0.0f);

			printf("[MAC]   velocity grids initialized\n");
		}

		__device__ void SetDefault() {
			m_DefaultOutOfRangeValue = 0.0f;
			m_Size = { 10, 10, 10 };
			m_DX = 0.1f;
			m_ExtrapolationLayerCount = 0;

			m_U.SetDefault();
			m_V.SetDefault();
			m_W.SetDefault();
		}
	private:
		Array3D<float> m_U;
		Array3D<float> m_V;
		Array3D<float> m_W;

		glm::ivec3 m_Size;
		float m_DX;
		float m_DefaultOutOfRangeValue;

		int m_ExtrapolationLayerCount;
	};
}

#endif // !MARKER_AND_CELL_VELOCITY_FIELD_CUH