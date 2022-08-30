#include "MeshLevelSet.cuh"
#include "pch.h"

namespace fe {
	__device__ Array3D<float> d_SDFPhi;
	__device__ Array3D<int> d_SDFClosestTriangles;
	__device__ Array3D<int> d_SDFIntersectionCounts;
	__device__ Array3D<bool> d_SDFSearchGrid;
	__device__ int d_SDFQueueIndex;

	template <typename T>
	__device__ T NewClamp(const T& n, const T& lower, const T& upper) {
		return max(lower, min(n, upper));
	}

	__device__ __host__ glm::vec3 NewGridIndexToPosition(int i, int j, int k, float dx) {
		return { i * dx, j * dx, k * dx };
	}

	__device__ __host__ float AccurateLength(const glm::vec3 v) {

		return sqrtl(v.x * v.x + v.y * v.y);
	}

	__device__ __host__ float AccurateLength2(const glm::vec3 v) {
		float l = AccurateLength(v);
		return l * l;
	}

	__device__ __host__ float NewPointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2) {
		glm::vec3 dx = x2 - x1;
		float m2 = AccurateLength2(dx);
		float s12 = glm::dot(x2 - x0, dx) / m2;
		if (s12 < 0) {
			s12 = 0;
		}
		else if (s12 > 1) {
			s12 = 1;
		}

		return AccurateLength(x0 - (s12 * x1 + (+-s12) * x2));
	}

	__device__ __host__ float NewPointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3) {
		glm::vec3 x13 = x1 - x3;
		glm::vec3 x23 = x2 - x3;
		glm::vec3 x03 = x0 - x3;

		float m13 = AccurateLength2(x13);
		float m23 = AccurateLength2(x23);
		float d = glm::dot(x13, x23);
		float invdet = 1.0f / fmax(m13 * m23 - d * d, 1e-30f);
		float a = glm::dot(x13, x03);
		float b = glm::dot(x23, x03);

		float w23 = invdet * (m23 * a - d * b);
		float w31 = invdet * (m13 * b - d * a);
		float w12 = 1 - w23 - w31;

		if (w23 >= 0 && w31 >= 0 && w12 >= 0) {
			return AccurateLength(x0 - (w23 * x1 + w31 * x2 + w12 * x3));
		}
		else {
			if (w23 > 0) {
				float d1 = NewPointToSegmentDistance(x0, x1, x2);
				float d2 = NewPointToSegmentDistance(x0, x1, x3);
				return fmin(d1, d2);
			}
			else if (w31 > 0) {
				// this rules out edge 1-3
				float d1 = NewPointToSegmentDistance(x0, x1, x2);
				float d2 = NewPointToSegmentDistance(x0, x2, x3);
				return fmin(d1, d2);
			}
			else {
				// w12 must be >0, ruling out edge 1-2
				float d1 = NewPointToSegmentDistance(x0, x1, x3);
				float d2 = NewPointToSegmentDistance(x0, x2, x3);
				return fmin(d1, d2);
			}
		}
	}

	__device__ int OrientationNew(float x1, float y1, float x2, float y2, float* twiceSignedArea) {
		*twiceSignedArea = y1 * x2 - x1 * y2;
		if (*twiceSignedArea > 0) {
			return 1;
		}
		else if (*twiceSignedArea < 0) {
			return -1;
		}
		else if (y2 > y1) {
			return 1;
		}
		else if (y2 < y1) {
			return -1;
		}
		else if (x1 > x2) {
			return 1;
		}
		else if (x1 < x2) {
			return -1;
		}
		else {
			return 0; // only true when x1==x2 and y1==y2
		}
	}

	__device__ bool GetBarycentricCoordinatesNew(
		float x0, float y0,
		float x1, float y1, float x2, float y2, float x3, float y3,
		float* a, float* b, float* c
	) {
		x1 -= x0;
		x2 -= x0;
		x3 -= x0;
		y1 -= y0;
		y2 -= y0;
		y3 -= y0;

		float oa;
		int signa = OrientationNew(x2, y2, x3, y3, &oa);
		if (signa == 0) {
			return false;
		}

		float ob;
		int signb = OrientationNew(x3, y3, x1, y1, &ob);
		if (signb != signa) {
			return false;
		}

		float oc;
		int signc = OrientationNew(x1, y1, x2, y2, &oc);
		if (signc != signa) {
			return false;
		}

		float sum = oa + ob + oc;
		assert(sum != 0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
		float invsum = 1.0f / sum;

		*a = oa * invsum;
		*b = ob * invsum;
		*c = oc * invsum;

		return true;
	}

	static __global__ void ComputeExactBandDistanceFieldKernel(int bandWidth, float DX, glm::ivec3 size, const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount) {
		const int index = blockIdx.x * blockDim.x + threadIdx.x;

		float invDX = 1.0f / DX;

		glm::ivec3 t = triangles[index];

		glm::vec3 p = vertices[t.x];
		glm::vec3 q = vertices[t.y];
		glm::vec3 r = vertices[t.z];

		float fip = p.x * invDX;
		float fjp = p.y * invDX;
		float fkp = p.z * invDX;

		float fiq = q.x * invDX;
		float fjq = q.y * invDX;
		float fkq = q.z * invDX;

		float fir = r.x * invDX;
		float fjr = r.y * invDX;
		float fkr = r.z * invDX;

		int i0 = glm::clamp(int(min(fip, min(fiq, fir))) - bandWidth, 0, size.x - 1);
		int j0 = glm::clamp(int(min(fjp, min(fjq, fjr))) - bandWidth, 0, size.y - 1);
		int k0 = glm::clamp(int(min(fkp, min(fkq, fkr))) - bandWidth, 0, size.z - 1);

		int i1 = glm::clamp(int(max(fip, max(fiq, fir))) + bandWidth + 1, 0, size.x - 1);
		int j1 = glm::clamp(int(max(fjp, max(fjq, fjr))) + bandWidth + 1, 0, size.y - 1);
		int k1 = glm::clamp(int(max(fkp, max(fkq, fkr))) + bandWidth + 1, 0, size.z - 1);

		for (int k = k0; k <= k1; k++) {
			for (int j = j0; j <= j1; j++) {
				for (int i = i0; i <= i1; i++) {
					glm::vec3 gPos = NewGridIndexToPosition(i, j, k, DX);
					float d = NewPointToTriangleDistance(gPos, p, q, r);

					if (d < d_SDFPhi(i, j, k)) {
						d_SDFPhi.Set(i, j, k, d);
						d_SDFClosestTriangles.Set(i, j, k, index);
					}
				}
			}
		}

		// Intersection counts
		j0 = clamp((int)ceil(min(fjp, min(fjq, fjr))), 0, size.y - 1);
		k0 = clamp((int)ceil(min(fkp, min(fkq, fkr))), 0, size.z - 1);

		j1 = clamp((int)floor(max(fjp, max(fjq, fjr))), 0, size.y - 1);
		k1 = clamp((int)floor(max(fkp, max(fkq, fkr))), 0, size.z - 1);

		for (int k = k0; k <= k1; k++) {
			for (int j = j0; j <= j1; j++) {
				float a, b, c;

				if (GetBarycentricCoordinatesNew(j, k, fjp, fkp, fjq, fkq, fjr, fkr, &a, &b, &c)) {
					float fi = a * fip + b * fiq + c * fir;
					int interval = int(ceil(fi));
					if (interval < 0) {
						d_SDFIntersectionCounts.Add(0, j, k, 1);
					}
					else if (interval < size.x) {
						d_SDFIntersectionCounts.Add(interval, j, k, 1);
					}
				}
			}
		}
	}

	static __host__ void NewGetNeighbourGridIndices6(const glm::ivec3 g, glm::ivec3 n[6]) {
		n[0] = { g.x - 1, g.y, g.z };
		n[1] = { g.x + 1, g.y, g.z };
		n[2] = { g.x, g.y - 1, g.z };
		n[3] = { g.x, g.y + 1, g.z };
		n[4] = { g.x, g.y, g.z - 1 };
		n[5] = { g.x, g.y, g.z + 1 };
	}

	static __host__ bool NewIsGridIndexInRange(const glm::ivec3 g, int imax, int jmax, int kmax) {
		return g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < imax&& g.y < jmax&& g.z < kmax;
	}

	__host__ void MeshLevelSet::CalculateSDFN(const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount, int bandWidth)
	{
		MeshVertices = vertices;
		MeshVertexCount = vertexCount;
		MeshTriangles = triangles;
		MeshTriangleCount = triangleCount;

		glm::ivec3 size = Phi.Size;

		Array3D<int> intersectionCounts;
		intersectionCounts.Init(size.x, size.y, size.z);

		Phi.Fill((size.x + size.y + size.z) * DX);
		ClosestTriangles.Fill(-1);
		intersectionCounts.Fill(0);

		Array3D<float> phiDevice;
		Array3D<int> closestTrianglesDevice;
		Array3D<int> intersectionCountsDevice;

		glm::vec3* meshVerticesDevice;
		glm::ivec3* meshTrianglesDevice;

		{
			auto start = std::chrono::high_resolution_clock::now();

			Phi.UploadToDevice(phiDevice, d_SDFPhi);
			ClosestTriangles.UploadToDevice(closestTrianglesDevice, d_SDFClosestTriangles);
			intersectionCounts.UploadToDevice(intersectionCountsDevice, d_SDFIntersectionCounts);


			cudaMalloc(&(meshVerticesDevice), sizeof(float) * 3 * vertexCount);
			cudaMemcpy(meshVerticesDevice, MeshVertices, sizeof(float) * 3 * vertexCount, cudaMemcpyHostToDevice);

			cudaMalloc(&(meshTrianglesDevice), sizeof(int) * 3 * triangleCount);
			cudaMemcpy(meshTrianglesDevice, MeshTriangles, sizeof(int) * 3 * triangleCount, cudaMemcpyHostToDevice);

			int threadCount;
			int blockCount;
			ComputeGridSize(MeshTriangleCount, 128, blockCount, threadCount);
			ComputeExactBandDistanceFieldKernel << < blockCount, threadCount >> > (bandWidth, DX, size, meshVerticesDevice, MeshVertexCount, meshTrianglesDevice, MeshTriangleCount);
			COMPUTE_SAFE(cudaDeviceSynchronize());

			auto stop = std::chrono::high_resolution_clock::now();
			LOG(std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) + " ms", ConsoleColor::Blue);
		}

		{
			auto start = std::chrono::high_resolution_clock::now();

			phiDevice.UploadToHost(Phi);
			intersectionCountsDevice.UploadToHost(intersectionCounts);
			closestTrianglesDevice.UploadToHost(ClosestTriangles);

			for (int k = 0; k < size.z; k++) {
				for (int j = 0; j < size.y; j++) {
					int tcount = 0;
					for (int i = 0; i < size.x; i++) {
						tcount += intersectionCounts(i, j, k);
						if (tcount % 2 == 1) {
							Phi.Set(i, j, k, -Phi(i, j, k));
						}
					}
				}
			}

			phiDevice.Free();
			intersectionCountsDevice.Free();
			closestTrianglesDevice.Free();

			delete[] intersectionCounts.Grid;

			COMPUTE_SAFE(cudaFree((void**)meshVerticesDevice));
			COMPUTE_SAFE(cudaFree((void**)meshTrianglesDevice));

			auto stop = std::chrono::high_resolution_clock::now();
			LOG(std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) + " ms", ConsoleColor::Blue);
		}
	}
}