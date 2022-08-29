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

	__device__ __host__ float NewPointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2) {
		glm::vec3 dx = x2 - x1;
		float m2 = glm::length2(dx);
		float s12 = glm::dot(x2 - x0, dx) / m2;
		if (s12 < 0) {
			s12 = 0;
		}
		else if (s12 > 1) {
			s12 = 1;
		}

		return glm::length(x0 - (s12 * x1 + (+-s12) * x2));
	}

	__device__ __host__ float NewPointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3) {
		glm::vec3 x13 = x1 - x3;
		glm::vec3 x23 = x2 - x3;
		glm::vec3 x03 = x0 - x3;

		float m13 = glm::length2(x13);
		float m23 = glm::length2(x23);
		float d = glm::dot(x13, x23);
		float invdet = 1.0f / fmax(m13 * m23 - d * d, 1e-30f);
		float a = glm::dot(x13, x03);
		float b = glm::dot(x23, x03);

		float w23 = invdet * (m23 * a - d * b);
		float w31 = invdet * (m13 * b - d * a);
		float w12 = 1 - w23 - w31;

		if (w23 >= 0 && w31 >= 0 && w12 >= 0) {
			return glm::length(x0 - (w23 * x1 + w31 * x2 + w12 * x3));
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

		double sum = oa + ob + oc;
		assert(sum != 0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
		double invsum = 1.0 / sum;

		*a = oa * invsum;
		*b = ob * invsum;
		*c = oc * invsum;

		return true;
	}

	static __global__ void ComputeExactBandDistanceFieldKernel(int bandWidth, float DX, glm::ivec3 size, const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount) {
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		const float invDX = 1.0f / DX;

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

		int i0 = NewClamp(int(fmin(fip, fmin(fiq, fir))) - bandWidth, 0, size.x - 1);
		int j0 = NewClamp(int(fmin(fjp, fmin(fjq, fjr))) - bandWidth, 0, size.y - 1);
		int k0 = NewClamp(int(fmin(fkp, fmin(fkq, fkr))) - bandWidth, 0, size.z - 1);

		int i1 = NewClamp(int(fmax(fip, fmax(fiq, fir))) + bandWidth + 1, 0, size.x - 1);
		int j1 = NewClamp(int(fmax(fjp, fmax(fjq, fjr))) + bandWidth + 1, 0, size.y - 1);
		int k1 = NewClamp(int(fmax(fkp, fmax(fkq, fkr))) + bandWidth + 1, 0, size.z - 1);

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
		j0 = NewClamp((int)std::ceil(fmin(fjp, fmin(fjq, fjr))), 0, size.y - 1);
		k0 = NewClamp((int)std::ceil(fmin(fkp, fmin(fkq, fkr))), 0, size.z - 1);

		j1 = NewClamp((int)std::floor(fmax(fjp, fmax(fjq, fjr))), 0, size.y - 1);
		k1 = NewClamp((int)std::floor(fmax(fkp, fmax(fkq, fkr))), 0, size.z - 1);

		for (int k = k0; k <= k1; k++) {
			for (int j = j0; j <= j1; j++) {
				float a, b, c;
				if (GetBarycentricCoordinatesNew(j, k, fjp, fkp, fjq, fkq, fjr, fkr, &a, &b, &c)) {
					float fi = a * fip + b * fiq + c * fir;
					int interval = int(ceil(fi));
					if (interval < 0) {
						d_SDFIntersectionCounts.AtomicAdd(0, j, k, 1);
					}
					else if (interval < size.x) {
						d_SDFIntersectionCounts.AtomicAdd(interval, j, k, 1);
					}
				}
			}
		}
	}

	static __global__ void PropagateKernel(glm::ivec3 size, glm::ivec3* queue) {
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

		for (int j = 0; j < size.y; j++) {
			for (int i = 0; i < size.x; i++) {
				if (d_SDFClosestTriangles(i, j, index) != -1) {
					d_SDFSearchGrid.Set(i, j, index, true);
					queue[d_SDFQueueIndex] = { i, j, index };
					atomicAdd(&d_SDFQueueIndex, 1);
				}
			}
		}
	}

	static __global__ void ComputeDistanceFieldSignsKernel(glm::ivec3 size) {
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

		for (int j = 0; j < size.y; j++) {
			int tcount = 0;
			for (int i = 0; i < size.x; i++) {
				tcount += d_SDFIntersectionCounts(i, j, index);
				if (tcount % 2 == 1) {
					d_SDFPhi.Set(i, j, index, -d_SDFPhi(i, j, index));
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

	__host__ void MeshLevelSet::CalculateSDFNew(const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount, int bandWidth)
	{
		MeshVertices = vertices;
		MeshVertexCount = vertexCount;
		MeshTriangles = triangles;
		MeshTriangleCount = triangleCount;

		glm::vec3 size = Phi.Size;
		glm::ivec3* queue = new glm::ivec3[size.x * size.y * size.z];

		Array3D<int> intersectionCounts;
		Array3D<bool> searchGrid;

		intersectionCounts.Init(size.x, size.y, size.z);
		searchGrid.Init(size.x, size.y, size.z, false);
		intersectionCounts.Fill(0);
		Phi.Fill((size.x + size.y + size.z) * DX);
		ClosestTriangles.Fill(-1);

		// Upload to the GPU 
		Array3D<float> phiDevice;
		Array3D<int> closestTrianglesDevice;
		Array3D<int> intersectionCountsDevice;
		Array3D<bool> searchGridDevice;

		// 	COMPUTE_SAFE(cudaMemcpyToSymbol(d_SDFQueueIndex, 0, sizeof(int)));

		Phi.UploadToDevice(phiDevice, d_SDFPhi);
		ClosestTriangles.UploadToDevice(closestTrianglesDevice, d_SDFClosestTriangles);
		intersectionCounts.UploadToDevice(intersectionCountsDevice, d_SDFIntersectionCounts);
		searchGrid.UploadToDevice(searchGridDevice, d_SDFSearchGrid);

		glm::vec3* meshVerticesDevice;
		glm::ivec3* meshTrianglesDevice;
		glm::ivec3* queueDevice;

		// Vertices
		cudaMalloc(&(meshVerticesDevice), sizeof(float) * 3 * vertexCount);
		cudaMemcpy(meshVerticesDevice, MeshVertices, sizeof(float) * 3 * vertexCount, cudaMemcpyHostToDevice);

		// Triangles
		cudaMalloc(&(meshTrianglesDevice), sizeof(int) * 3 * triangleCount);
		cudaMemcpy(meshTrianglesDevice, MeshTriangles, sizeof(int) * 3 * triangleCount, cudaMemcpyHostToDevice);

		// Queue
		cudaMalloc(&(queueDevice), sizeof(int) * 3 * size.x * size.y * size.z );

		// Initialize distances near the mesh 
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(MeshTriangleCount, 32, blockCount, threadCount);

			ComputeExactBandDistanceFieldKernel <<< blockCount, threadCount >>> (bandWidth, DX, size, meshVerticesDevice, MeshVertexCount, meshTrianglesDevice, MeshTriangleCount);
			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		// Propagate distances outwards
		{
			PropagateKernel <<< 1, size.z >>> (size, queueDevice);
			COMPUTE_SAFE(cudaDeviceSynchronize());

			int queueIndex;
			COMPUTE_SAFE(cudaMemcpyFromSymbol(&queueIndex, d_SDFQueueIndex, sizeof(int)));
			int queueSize = queueIndex;

			int unknownIdx = queueIndex;
			int startIdx = 0;

			// Upload back to the CPU
		    cudaMemcpy(queue, queueDevice, sizeof(int) * 3 * size.x * size.y * size.z, cudaMemcpyDeviceToHost); // TODO: use queueSize * element size
			searchGridDevice.UploadToHost(searchGrid);
			closestTrianglesDevice.UploadToHost(ClosestTriangles);
			phiDevice.UploadToHost(Phi);

			glm::ivec3 g, n, nbs[6];
			while (startIdx < queueSize) {
				g = queue[startIdx];
				startIdx++;

				NewGetNeighbourGridIndices6(g, nbs);
				for (int nIdx = 0; nIdx < 6; nIdx++) {
					n = nbs[nIdx];
					if (NewIsGridIndexInRange(n, size.x, size.y, size.z) && searchGrid(n) == false) {
						searchGrid.Set(n, true);
						queue[queueIndex] = n;
						queueIndex++;
					}
				}
			}

			glm::vec3 gPos;
			glm::ivec3 t;
			startIdx = unknownIdx;
			queueSize = queueIndex;
			while (startIdx < queueSize) {
				g = queue[startIdx];
				startIdx++;

				gPos = NewGridIndexToPosition(g.x, g.y, g.z, DX);
				NewGetNeighbourGridIndices6(g, nbs);
				for (int nIdx = 0; nIdx < 6; nIdx++) {
					n = nbs[nIdx];
					if (NewIsGridIndexInRange(n, size.x, size.y, size.z) && ClosestTriangles(n) != -1) {
						t = MeshTriangles[ClosestTriangles(n)];
						float dist = NewPointToTriangleDistance(gPos, MeshVertices[t.x], MeshVertices[t.y], MeshVertices[t.z]);
						if (dist < Phi(g)) {
							Phi.Set(g, dist);
							ClosestTriangles.Set(g, ClosestTriangles(n));
						}
					}
				}
			}
		}

		// Figure out signs (inside / outside) from intersection counts
		{
			Phi.UploadToDevice(phiDevice, d_SDFPhi);

			ComputeDistanceFieldSignsKernel << < 1, size.z >> > (size);
			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		// Upload back to the CPU
		phiDevice.UploadToHost(Phi);

		// Free GPU memory
		phiDevice.Free();
		closestTrianglesDevice.Free();
		intersectionCountsDevice.Free();
		searchGridDevice.Free();

		delete[] intersectionCounts.Grid;
		delete[] searchGrid.Grid;
		delete[] queue;

		COMPUTE_SAFE(cudaFree((void**)meshVerticesDevice));
		COMPUTE_SAFE(cudaFree((void**)meshTrianglesDevice));
		COMPUTE_SAFE(cudaFree((void**)queueDevice));
	}
}