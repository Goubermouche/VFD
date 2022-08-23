#ifndef MARKER_AND_CELL_VELOCITY_FIELD_CUH
#define MARKER_AND_CELL_VELOCITY_FIELD_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"

namespace fe {
	struct ValidVelocityComponentGrid {
		Array3D<bool> ValidU;
		Array3D<bool> ValidV;
		Array3D<bool> ValidW;

		__device__ ValidVelocityComponentGrid() = default;
		__device__ ValidVelocityComponentGrid(int i, int j, int k) 
			: ValidU(i + 1, j, k, false), ValidV(i, j + 1, k, false), ValidW(i, j, k + 1, false)
		{}

		__device__ void Reset() {
			ValidU.Fill(false);
			ValidV.Fill(false);
			ValidW.Fill(false);
		}
	};

	struct TestStruct {
		__device__ TestStruct() {}

		__device__ TestStruct(int i)
			: value(i)
		{}

		glm::vec3 v;
		int value;
	};

	struct MACVelocityField {
		__device__ MACVelocityField() = default;

		__device__ MACVelocityField(int i, int j, int k, float dx)
			: m_Size(i, j, k), m_DX(dx) {
			InitializeVelocityGrids();
		}

		__device__ ~MACVelocityField() = default;
	private:
		__device__ void InitializeVelocityGrids() {
			m_U = Array3D<float>(m_Size.x + 1, m_Size.y, m_Size.z, 0.0f);
			m_V = Array3D<float>(m_Size.x, m_Size.y + 1, m_Size.z, 0.0f);
			m_W = Array3D<float>(m_Size.x, m_Size.y, m_Size.z + 1, 0.0f);

			m_U.SetOutOfRangeValue(0.0f);
			m_V.SetOutOfRangeValue(0.0f);
			m_W.SetOutOfRangeValue(0.0f);
		}

		glm::ivec3 m_Size = { 10, 10, 10 };
		float m_DX = 0.1f;

		Array3D<float> m_U;
		Array3D<float> m_V;
		Array3D<float> m_W;

		int m_ExtrapolationLayerCount = 0;
	};
}

#endif // !MARKER_AND_CELL_VELOCITY_FIELD_CUH