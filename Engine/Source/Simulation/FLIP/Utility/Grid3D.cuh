#ifndef GRID_3D_CUH
#define GRID_3D_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/Array3D.cuh"

namespace fe {
	static __device__ __host__ glm::vec3 GridIndexToPosition(int i, int j, int k, double dx) {
		return { i * dx, j * dx, k * dx };
	}


    static __device__ __host__ void GridIndexToPosition(int i, int j, int k, double dx,
        double* x, double* y, double* z) {
        *x = (double)i * dx;
        *y = (double)j * dx;
        *z = (double)k * dx;
    }

    static __device__ __host__ glm::vec3 GridIndexToPosition(glm::ivec3 g, double dx) {
        return glm::vec3((float)(g.x * dx), (float)(g.y * dx), (float)(g.z * dx));
    }

    static __device__ __host__ bool IsGridIndexInRange(int i, int j, int k, int imax, int jmax, int kmax) {
        return i >= 0 && j >= 0 && k >= 0 && i < imax&& j < jmax&& k < kmax;
    }

	static __device__ __host__ glm::ivec3 PositionToGridIndex(const glm::vec3& p , double dx) {
        double invdx = 1.0 / dx;
        return glm::ivec3((int)floor(p.x * invdx),
            (int)floor(p.y * invdx),
            (int)floor(p.z * invdx));
	}

    static __device__ __host__ void PositionToGridIndex(double x, double y, double z, double dx,
        int* i, int* j, int* k) {
        double invdx = 1.0 / dx;
        *i = (int)floor(x * invdx);
        *j = (int)floor(y * invdx);
        *k = (int)floor(z * invdx);
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

	static __device__ __host__ glm::vec3 GridIndexToCellCenter(int i, int j, int k, double dx) {
        double hw = 0.5 * dx;
        return glm::vec3((float)(i * dx + hw), (float)(j * dx + hw), (float)(k * dx + hw));
	}

    template <class T>
    __device__ __host__ bool IsFaceBorderingValueU(int i, int j, int k, T m, Array3D<T>& grid) {
        if (i == grid.Size.x) { return grid(i - 1, j, k) == m; }
        else if (i > 0) { return grid(i, j, k) == m || grid(i - 1, j, k) == m; }
        else { return grid(i, j, k) == m; }
    }

    template <class T>
    static __device__ __host__ bool IsFaceBorderingValueU(glm::ivec3 g, T m, Array3D<T>& grid) {
        return isFaceBorderingValueU(g.x, g.y, g.z, m, grid);
    }

    template <class T>
    static __device__ __host__ bool IsFaceBorderingValueV(int i, int j, int k, T m, Array3D<T>& grid) {
        if (j == grid.Size.y) { return grid(i, j - 1, k) == m; }
        else if (j > 0) { return grid(i, j, k) == m || grid(i, j - 1, k) == m; }
        else { return grid(i, j, k) == m; }
    }

    template <class T>
    static  __device__ __host__ bool IsFaceBorderingValueV(glm::ivec3 g, T m, Array3D<T>& grid) {
        return isFaceBorderingValueV(g.x, g.y, g.z, m, grid);
    }

    template <class T>
    static __device__ __host__ bool IsFaceBorderingValueW(int i, int j, int k, T m, Array3D<T>& grid) {
        if (k == grid.Size.z) { return grid(i, j, k - 1) == m; }
        else if (k > 0) { return grid(i, j, k) == m || grid(i, j, k - 1) == m; }
        else { return grid(i, j, k) == m; }
    }

    template <class T>
    static __device__ __host__ bool IsFaceBorderingValueW(glm::ivec3 g, T m, Array3D<T>& grid) {
        return isFaceBorderingValueW(g.x, g.y, g.z, m, grid);
    }

    static __device__ __host__ bool IsGridIndexOnBorder(int i, int j, int k, int imax, int jmax, int kmax) {
        return i == 0 || j == 0 || k == 0 ||
            i == imax - 1 || j == jmax - 1 || k == kmax - 1;
    }

    static __device__ __host__ bool IsPositionInGrid(double x, double y, double z, double dx, int i, int j, int k) {
        return x >= 0 && y >= 0 && z >= 0 && x < dx* i&& y < dx* j&& z < dx* k;
    }
}


#endif // !GRID_3D_CUH