#ifndef GRID_3D_CUH
#define GRID_3D_CUH

#include "pch.h"

namespace fe {
	static __device__ __host__ glm::vec3 GridIndexToPosition(int i, int j, int k, float dx) {
		return { i * dx, j * dx, k * dx };
	}

	static __device__ __host__ glm::ivec3 PositionToGridIndex(const glm::vec3& p , float dx) {
		float invDx = 1.0f / dx;
		return { (int)floor(p.x * invDx),
			(int)floor(p.y * invDx),
			(int)floor(p.z * invDx) };
	}

	static __device__ __host__ void GetNeighbourGridIndices6(const glm::ivec3 g, glm::ivec3 n[6]) {
		n[0] = {g.x - 1, g.y, g.z};
		n[1] = {g.x + 1, g.y, g.z};
		n[2] = {g.x, g.y - 1, g.z};
		n[3] = {g.x, g.y + 1, g.z};
		n[4] = {g.x, g.y, g.z - 1};
		n[5] = {g.x, g.y, g.z + 1};
	}

	static __device__ __host__ bool IsGridIndexInRange(const glm::ivec3 g, int imax, int jmax, int kmax) {
		return g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < imax&& g.y < jmax&& g.z < kmax;
	}
}

#endif // !GRID_3D_CUH