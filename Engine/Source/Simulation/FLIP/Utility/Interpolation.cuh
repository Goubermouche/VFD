#ifndef INTERPOLATION_CUH
#define INTERPOLATION_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/Grid3d.cuh"

namespace fe {
	struct Interpolation {
		static __device__ __host__ float TrilinearInterpolate(const glm::vec3& p, float dx, Array3D<float>& grid) {
			glm::ivec3 g = PositionToGridIndex(p, dx);
			glm::vec3 gpos = GridIndexToPosition(g.x, g.y, g.z, dx);

			float inv_dx = 1.0 / dx;
			float ix = (p.x - gpos.x) * inv_dx;
			float iy = (p.y - gpos.y) * inv_dx;
			float iz = (p.z - gpos.z) * inv_dx;

			float points[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			int isize = grid.Size.x;
			int jsize = grid.Size.y;
			int ksize = grid.Size.z;

            if (IsGridIndexInRange(glm::ivec3{g.x, g.y, g.z }, isize, jsize, ksize)) {
                points[0] = grid(g.x, g.y, g.z);
            }
            if (IsGridIndexInRange({g.x + 1, g.y, g.z }, isize, jsize, ksize)) {
                points[1] = grid(g.x + 1, g.y, g.z);
            }
            if (IsGridIndexInRange({g.x, g.y + 1, g.z }, isize, jsize, ksize)) {
                points[2] = grid(g.x, g.y + 1, g.z);
            }
            if (IsGridIndexInRange({g.x, g.y, g.z + 1 }, isize, jsize, ksize)) {
                points[3] = grid(g.x, g.y, g.z + 1);
            }
            if (IsGridIndexInRange({g.x + 1, g.y, g.z + 1 }, isize, jsize, ksize)) {
                points[4] = grid(g.x + 1, g.y, g.z + 1);
            }
            if (IsGridIndexInRange({g.x, g.y + 1, g.z + 1 }, isize, jsize, ksize)) {
                points[5] = grid(g.x, g.y + 1, g.z + 1);
            }
            if (IsGridIndexInRange({g.x + 1, g.y + 1, g.z }, isize, jsize, ksize)) {
                points[6] = grid(g.x + 1, g.y + 1, g.z);
            }
            if (IsGridIndexInRange({g.x + 1, g.y + 1, g.z + 1}, isize, jsize, ksize)) {
                points[7] = grid(g.x + 1, g.y + 1, g.z + 1);
            }

            return TrilinearInterpolate(points, ix, iy, iz);
		}

        static __device__ __host__ float TrilinearInterpolate(float p[8], float x, float y, float z) {
            return p[0] * (1 - x) * (1 - y) * (1 - z) +
                p[1] * x * (1 - y) * (1 - z) +
                p[2] * (1 - x) * y * (1 - z) +
                p[3] * (1 - x) * (1 - y) * z +
                p[4] * x * (1 - y) * z +
                p[5] * (1 - x) * y * z +
                p[6] * x * y * (1 - z) +
                p[7] * x * y * z;
        }
	};
}

#endif // !INTERPOLATION_CUH
