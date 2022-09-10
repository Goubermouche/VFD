#include "MeshLevelSet.cuh"
#include "pch.h"

#include "Simulation/FLIP/Utility/Grid3D.cuh"

namespace fe {
	__device__ Array3D<float> d_SDFPhi;
	__device__ Array3D<int> d_SDFClosestTriangles;
	__device__ Array3D<int> d_SDFIntersectionCounts;

	// TODO: move to a cuda math file 
	__host__ __device__ float AccurateLength(const glm::vec3 v) {
		return sqrtl(v.x * v.x + v.y * v.y + v.z * v.z);
	}

	// TODO: move to a cuda math file 
	__host__ __device__ float AccurateLength2(const glm::vec3 v) {
		float l = AccurateLength(v);
		return l * l;
	}

	__host__ __device__ float PointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2) {
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

	__host__ __device__ float PointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3) {
		glm::vec3 x13 = x1 - x3;
		glm::vec3 x23 = x2 - x3;
		glm::vec3 x03 = x0 - x3;

		float m13 = AccurateLength2(x13);
		float m23 = AccurateLength2(x23);
		float d = glm::dot(x13, x23);
		float invDet = 1.0f / fmax(m13 * m23 - d * d, 1e-30f);
		float a = glm::dot(x13, x03);
		float b = glm::dot(x23, x03);

		float w23 = invDet * (m23 * a - d * b);
		float w31 = invDet * (m13 * b - d * a);
		float w12 = 1 - w23 - w31;

		if (w23 >= 0.0f && w31 >= 0.0f && w12 >= 0.0f) {
			return AccurateLength(x0 - (w23 * x1 + w31 * x2 + w12 * x3));
		}
		else {
			if (w23 > 0.0f) {
				float d1 = PointToSegmentDistance(x0, x1, x2);
				float d2 = PointToSegmentDistance(x0, x1, x3);
				return min(d1, d2);
			}
			else if (w31 > 0.0f) {
				float d1 = PointToSegmentDistance(x0, x1, x2);
				float d2 = PointToSegmentDistance(x0, x2, x3);
				return min(d1, d2);
			}
			else {
				float d1 = PointToSegmentDistance(x0, x1, x3);
				float d2 = PointToSegmentDistance(x0, x2, x3);
				return min(d1, d2);
			}
		}
	}

	__host__ __device__ int Orientation(double x1, double y1, double x2, double y2, double* twiceSignedArea) {
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
			return 0;
		}
	}

	__host__ __device__ bool GetBarycentricCoordinates(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3, double* a, double* b, double* c) {
		x1 -= x0;
		x2 -= x0;
		x3 -= x0;
		y1 -= y0;
		y2 -= y0;
		y3 -= y0;

		double oa;
		int signA = Orientation(x2, y2, x3, y3, &oa);
		if (signA == 0) {
			return false;
		}

		double ob;
		int signB = Orientation(x3, y3, x1, y1, &ob);
		if (signB != signA) {
			return false;
		}

		double oc;
		int signC = Orientation(x1, y1, x2, y2, &oc);
		if (signC != signA) {
			return false;
		}

		double sum = oa + ob + oc;
		double invSum = 1.0 / sum;

		*a = oa * invSum;
		*b = ob * invSum;
		*c = oc * invSum;

		return true;
	}

	__global__ void CalculateExactBandDistanceFieldKernel(int bandWidth, double DX, double invDX, glm::ivec3 size, const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount) {
		const int Indices = blockIdx.x * blockDim.x + threadIdx.x;

		glm::ivec3 t = triangles[Indices];

		glm::vec3 p = vertices[t.x];
		glm::vec3 q = vertices[t.y];
		glm::vec3 r = vertices[t.z];

		double fip = (double)p.x * invDX;
		double fjp = (double)p.y * invDX;
		double fkp = (double)p.z * invDX;

		double fiq = (double)q.x * invDX;
		double fjq = (double)q.y * invDX;
		double fkq = (double)q.z * invDX;

		double fir = (double)r.x * invDX;
		double fjr = (double)r.y * invDX;
		double fkr = (double)r.z * invDX;

		int i0 = clamp(int(min(fip, min(fiq, fir))) - bandWidth, 0, size.x - 1);
		int j0 = clamp(int(min(fjp, min(fjq, fjr))) - bandWidth, 0, size.y - 1);
		int k0 = clamp(int(min(fkp, min(fkq, fkr))) - bandWidth, 0, size.z - 1);
		int i1 = clamp(int(max(fip, max(fiq, fir))) + bandWidth + 1, 0, size.x - 1);
		int j1 = clamp(int(max(fjp, max(fjq, fjr))) + bandWidth + 1, 0, size.y - 1);
		int k1 = clamp(int(max(fkp, max(fkq, fkr))) + bandWidth + 1, 0, size.z - 1);

		for (int k = k0; k <= k1; k++) {
			for (int j = j0; j <= j1; j++) {
				for (int i = i0; i <= i1; i++) {
					glm::vec3 pos = GridIndexToPosition(i, j, k, DX);
					float d = PointToTriangleDistance(pos, p, q, r);

					if (d < d_SDFPhi(i, j, k)) {
						d_SDFPhi.Set(i, j, k, d);
						d_SDFClosestTriangles.Set(i, j, k, Indices);
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
				double a;
				double b;
				double c;

				if (GetBarycentricCoordinates(j, k, fjp, fkp, fjq, fkq, fjr, fkr, &a, &b, &c)) {
					double fi = a * fip + b * fiq + c * fir;
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

	__global__ void CalculateDistanceFieldSignsKernel(int sizeX) {
		int k = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int count = 0;

		for (int i = 0; i < sizeX; i++) {
			count += d_SDFIntersectionCounts(i, j, k);
			if (count % 2 == 1) {
				d_SDFPhi.Set(i, j, k, -d_SDFPhi(i, j, k));
			}
		}
	}

	__host__ void MeshLevelSet::Init(TriangleMesh& mesh, int resolution, double dx, int bandWidth)
	{
		const auto vertices = mesh.GetVertices();
		const auto triangles = mesh.GetTriangles();

		MeshVertexCount = vertices.size();
		MeshTriangleCount = triangles.size();

		MeshVertices = new glm::vec3[MeshVertexCount];
		MeshTriangles = new glm::ivec3[MeshTriangleCount];

		std::copy(vertices.data(), vertices.data() + MeshVertexCount, MeshVertices);
		std::copy(triangles.data(), triangles.data() + MeshTriangleCount, MeshTriangles);

		Size = { resolution, resolution, resolution };
		DX = dx;
		Phi.Init(resolution + 1, resolution + 1, resolution + 1, 0.0f);
		ClosestTriangles.Init(resolution + 1, resolution + 1, resolution + 1, -1);

		glm::ivec3 size = Phi.Size;

		Array3D<int> intersectionCounts;
		intersectionCounts.Init(size.x, size.y, size.z);

		Phi.Fill((size.x + size.y + size.z) * DX);
		ClosestTriangles.Fill(-1);
		intersectionCounts.Fill(0);

		Array3D<float> phiDevice;
		Array3D<int> closestTrianglesDevice;
		Array3D<int> intersectionCountsDevice;

		Phi.UploadToDevice(phiDevice, d_SDFPhi);
		ClosestTriangles.UploadToDevice(closestTrianglesDevice, d_SDFClosestTriangles);
		intersectionCounts.UploadToDevice(intersectionCountsDevice, d_SDFIntersectionCounts);

		glm::vec3* meshVerticesDevice;
		glm::ivec3* meshTrianglesDevice;

		cudaMalloc(&meshVerticesDevice, sizeof(glm::vec3) * MeshVertexCount);
		cudaMemcpy(meshVerticesDevice, MeshVertices, sizeof(glm::vec3) * MeshVertexCount, cudaMemcpyHostToDevice);

		cudaMalloc(&meshTrianglesDevice, sizeof(glm::ivec3) * MeshTriangleCount);
		cudaMemcpy(meshTrianglesDevice, MeshTriangles, sizeof(glm::ivec3) * MeshTriangleCount, cudaMemcpyHostToDevice);

		{
			int blockCount;
			int threadCount;
			ComputeGridSize(MeshTriangleCount, 128, blockCount, threadCount);
			CalculateExactBandDistanceFieldKernel <<< blockCount, threadCount >>> (bandWidth, DX, 1.0 / DX, size, meshVerticesDevice, MeshVertexCount, meshTrianglesDevice, MeshTriangleCount);
			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		{
			dim3 blockCount;
			dim3 threadCount;
			ComputeGridSize({ resolution, resolution }, { 128, 128 }, blockCount, threadCount);
			CalculateDistanceFieldSignsKernel <<< blockCount, threadCount >>> (size.x);
			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		phiDevice.UploadToHost(Phi);
		intersectionCountsDevice.UploadToHost(intersectionCounts);
		closestTrianglesDevice.UploadToHost(ClosestTriangles);

		phiDevice.DeviceFree();
		intersectionCountsDevice.DeviceFree();
		closestTrianglesDevice.DeviceFree();

		delete[] intersectionCounts.Grid;

		COMPUTE_SAFE(cudaFree((void**)meshVerticesDevice));
		COMPUTE_SAFE(cudaFree((void**)meshTrianglesDevice));
	}

	__host__ void MeshLevelSet::CalculateSDF(TriangleMesh& mesh, int resolution, double dx, int bandWidth)
	{
		const auto vertices = mesh.GetVertices();
		const auto triangles = mesh.GetTriangles();

		MeshVertexCount = vertices.size();
		MeshTriangleCount = triangles.size();

		MeshVertices = new glm::vec3[MeshVertexCount];
		MeshTriangles = new glm::ivec3[MeshTriangleCount];

		std::copy(vertices.data(), vertices.data() + MeshVertexCount, MeshVertices);
		std::copy(triangles.data(), triangles.data() + MeshTriangleCount, MeshTriangles);

		Size = { resolution, resolution, resolution };
		DX = dx;
		ClosestTriangles.Init(resolution + 1, resolution + 1, resolution + 1, -1);
		Phi.Init(resolution + 1, resolution + 1, resolution + 1, 0.0f);
		Array3D<int> intersectionCounts;
		intersectionCounts.Init(Phi.Size.x, Phi.Size.y, Phi.Size.z);


		{
			glm::ivec3 size = Phi.Size;
			Phi.Fill((size.x + size.y + size.z) * DX);
			ClosestTriangles.Fill(-1);
			intersectionCounts.Fill(0);
			glm::ivec3 t;
			float invDX = 1.0f / DX;

			for (size_t tIdx = 0; tIdx < MeshTriangleCount; tIdx++)
			{
				t = MeshTriangles[tIdx];
				glm::vec3 p = MeshVertices[t.x];
				glm::vec3 q = MeshVertices[t.y];
				glm::vec3 r = MeshVertices[t.z];

				float fip = p.x * invDX;
				float fjp = p.y * invDX;
				float fkp = p.z * invDX;

				float fiq = q.x * invDX;
				float fjq = q.y * invDX;
				float fkq = q.z * invDX;

				float fir = r.x * invDX;
				float fjr = r.y * invDX;
				float fkr = r.z * invDX;

				int i0 = clamp(int(fmin(fip, fmin(fiq, fir))) - bandWidth, 0, size.x - 1);
				int j0 = clamp(int(fmin(fjp, fmin(fjq, fjr))) - bandWidth, 0, size.y - 1);
				int k0 = clamp(int(fmin(fkp, fmin(fkq, fkr))) - bandWidth, 0, size.z - 1);

				int i1 = clamp(int(fmax(fip, fmax(fiq, fir))) + bandWidth + 1, 0, size.x - 1);
				int j1 = clamp(int(fmax(fjp, fmax(fjq, fjr))) + bandWidth + 1, 0, size.y - 1);
				int k1 = clamp(int(fmax(fkp, fmax(fkq, fkr))) + bandWidth + 1, 0, size.z - 1);

				for (int k = k0; k <= k1; k++) {
					for (int j = j0; j <= j1; j++) {
						for (int i = i0; i <= i1; i++) {
							glm::vec3 gPos = GridIndexToPosition(i, j, k, DX);
							float d = PointToTriangleDistance(gPos, p, q, r);
							if (d < Phi(i, j, k)) {
								Phi.Set(i, j, k, d);
								ClosestTriangles.Set(i, j, k, tIdx);
							}
						}
					}
				}

				// Intersection counts
				j0 = clamp((int)std::ceil(fmin(fjp, fmin(fjq, fjr))), 0, size.y - 1);
				k0 = clamp((int)std::ceil(fmin(fkp, fmin(fkq, fkr))), 0, size.z - 1);

				j1 = clamp((int)std::floor(fmax(fjp, fmax(fjq, fjr))), 0, size.y - 1);
				k1 = clamp((int)std::floor(fmax(fkp, fmax(fkq, fkr))), 0, size.z - 1);

				for (int k = k0; k <= k1; k++) {
					for (int j = j0; j <= j1; j++) {
						double a, b, c;
						if (GetBarycentricCoordinates(j, k, fjp, fkp, fjq, fkq, fjr, fkr, &a, &b, &c)) {
							double fi = a * fip + b * fiq + c * fir;
							int interval = int(ceil(fi));
							if (interval < 0) {
								intersectionCounts.Add(0, j, k, 1);
							}
							else if (interval < size.x) {
								intersectionCounts.Add(interval, j, k, 1);
							}
						}
					}
				}
			}
		}

		{
			glm::ivec3 size = Phi.Size;

			std::vector<glm::ivec3> queue;
			queue.reserve(size.x * size.y * size.z);
			Array3D<bool> searchGrid;
			searchGrid.Init(size.x, size.y, size.z, false);
			for (int k = 0; k < size.z; k++) {
				for (int j = 0; j < size.y; j++) {
					for (int i = 0; i < size.x; i++) {
						if (ClosestTriangles(i, j, k) != -1) {
							searchGrid.Set(i, j, k, true);
							queue.push_back({ i, j, k });
						}
					}
				}
			}

			int unknownIdx = queue.size();
			int startIdx = 0;
			glm::ivec3 g, Count, nbs[6];
			while (startIdx < (int)queue.size()) {
				g = queue[startIdx];
				startIdx++;

				GetNeighbourGridIndices6(g, nbs);
				for (int nIdx = 0; nIdx < 6; nIdx++) {
					Count = nbs[nIdx];
					if (IsGridIndexInRange(Count, size.x, size.y, size.z) && searchGrid(Count) == false) {
						searchGrid.Set(Count, true);
						queue.push_back(Count);
					}
				}
			}

			glm::vec3 gPos;
			glm::ivec3 t;
			startIdx = unknownIdx;
			while (startIdx < queue.size()) {
				g = queue[startIdx];
				startIdx++;

				gPos = GridIndexToPosition(g.x, g.y, g.z, DX);
				GetNeighbourGridIndices6(g, nbs);
				for (int nIdx = 0; nIdx < 6; nIdx++) {
					Count = nbs[nIdx];
					if (IsGridIndexInRange(Count, size.x, size.y, size.z) && ClosestTriangles(Count) != -1) {
						t = MeshTriangles[ClosestTriangles(Count)];
						float dist = PointToTriangleDistance(gPos, MeshVertices[t.x], MeshVertices[t.y], MeshVertices[t.z]);
						if (dist < Phi(g)) {
							Phi.Set(g, dist);
							ClosestTriangles.Set(g, ClosestTriangles(Count));
						}
					}
				}
			}
		}

		{
			glm::ivec3 size = Phi.Size;

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
		}

		intersectionCounts.HostFree();
	}
}