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

        static __device__ __host__ float BilinearInterpolate(
            float v00, float v10, float v01, float v11, float ix, float iy) {
            float lerp1 = (1 - ix) * v00 + ix * v10;
            float lerp2 = (1 - ix) * v01 + ix * v11;

            return (1 - iy) * lerp1 + iy * lerp2;
        }

        static __device__ __host__ void TrilinearInterpolateGradient(
            glm::vec3 p, float dx, Array3D<float>& grid, glm::vec3* grad) {

            glm::ivec3 g = PositionToGridIndex(p, dx);
            glm::vec3 gpos = GridIndexToPosition(g, dx);

            float inv_dx = 1.0 / dx;
            float ix = (p.x - gpos.x) * inv_dx;
            float iy = (p.y - gpos.y) * inv_dx;
            float iz = (p.z - gpos.z) * inv_dx;

            int isize = grid.Size.x;
            int jsize = grid.Size.y;
            int ksize = grid.Size.z;

            float v000 = 0, v001 = 0, v010 = 0, v011 = 0, v100 = 0, v101 = 0, v110 = 0, v111 = 0;
            if (IsGridIndexInRange(g.x, g.y, g.z, isize, jsize, ksize)) {
                v000 = grid(g.x, g.y, g.z);
            }
            if (IsGridIndexInRange(g.x + 1, g.y, g.z, isize, jsize, ksize)) {
                v100 = grid(g.x + 1, g.y, g.z);
            }
            if (IsGridIndexInRange(g.x, g.y + 1, g.z, isize, jsize, ksize)) {
                v010 = grid(g.x, g.y + 1, g.z);
            }
            if (IsGridIndexInRange(g.x, g.y, g.z + 1, isize, jsize, ksize)) {
                v001 = grid(g.x, g.y, g.z + 1);
            }
            if (IsGridIndexInRange(g.x + 1, g.y, g.z + 1, isize, jsize, ksize)) {
                v101 = grid(g.x + 1, g.y, g.z + 1);
            }
            if (IsGridIndexInRange(g.x, g.y + 1, g.z + 1, isize, jsize, ksize)) {
                v011 = grid(g.x, g.y + 1, g.z + 1);
            }
            if (IsGridIndexInRange(g.x + 1, g.y + 1, g.z, isize, jsize, ksize)) {
                v110 = grid(g.x + 1, g.y + 1, g.z);
            }
            if (IsGridIndexInRange(g.x + 1, g.y + 1, g.z + 1, isize, jsize, ksize)) {
                v111 = grid(g.x + 1, g.y + 1, g.z + 1);
            }

            float ddx00 = v100 - v000;
            float ddx10 = v110 - v010;
            float ddx01 = v101 - v001;
            float ddx11 = v111 - v011;
            float dv_dx = (float)BilinearInterpolate(ddx00, ddx10, ddx01, ddx11, iy, iz);

            float ddy00 = v010 - v000;
            float ddy10 = v110 - v100;
            float ddy01 = v011 - v001;
            float ddy11 = v111 - v101;
            float dv_dy = (float)BilinearInterpolate(ddy00, ddy10, ddy01, ddy11, ix, iz);

            float ddz00 = v001 - v000;
            float ddz10 = v101 - v100;
            float ddz01 = v011 - v010;
            float ddz11 = v111 - v110;
            float dv_dz = (float)BilinearInterpolate(ddz00, ddz10, ddz01, ddz11, ix, iy);

            grad->x = dv_dx;
            grad->y = dv_dy;
            grad->z = dv_dz;
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
