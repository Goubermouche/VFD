#ifndef MESH_LEVEL_SET_CUH 
#define MESH_LEVEL_SET_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/Grid3D.cuh"
#include "Compute/Utility/CUDA/cutil_math.h"
#include "Simulation/FLIP/Utility/Interpolation.cuh"
#include "Compute/Utility/CudaKernelUtility.cuh"

namespace fe {
	struct MeshLevelSet {
		__device__ void Init(int i, int j, int k, float dx) {
			Size = { i, j, k };
			DX = dx;
			Phi.Init(i + 1, j + 1, k + 1, 0.0f);
			ClosestTriangles.Init(i + 1, j + 1, k + 1, -1);
		}

		__device__ __host__ void CalculateSDF(const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount, int bandWidth = -1) {
			auto start = std::chrono::high_resolution_clock::now();

			MeshVertices = vertices;
			MeshVertexCount = vertexCount;
			MeshTriangles = triangles;
			MeshTriangleCount = triangleCount;

			Array3D<int> intersectionCounts;
			intersectionCounts.Init(Phi.Size.x, Phi.Size.y, Phi.Size.z);

			// Initialize distances near the mesh
			ComputeExactBandDistanceField(bandWidth, intersectionCounts);
			// Propagate distances outwards
			// PropagateDistanceField();

			// Figure out signs (inside / outside) from intersection counts
			ComputeDistanceFieldSigns(intersectionCounts);

			auto stop = std::chrono::high_resolution_clock::now();
			LOG("old: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) + " ms", ConsoleColor::Blue);
		}

		// TODO: convert into a kernel
		__device__ __host__ void ComputeExactBandDistanceField(int bandWidth, Array3D<int>& intersectionCounts) {
			glm::ivec3 size = Phi.Size;
			Phi.Fill((size.x + size.y + size.z) * DX);
			ClosestTriangles.Fill(-1);
			intersectionCounts.Fill(0);

			glm::ivec3 t;
			float invDX = 1.0f / DX;
			for (size_t tIdx= 0; tIdx < MeshTriangleCount; tIdx++)
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

				int i0 = glm::clamp(int(fmin(fip, fmin(fiq, fir))) - bandWidth, 0, size.x - 1);
				int j0 = glm::clamp(int(fmin(fjp, fmin(fjq, fjr))) - bandWidth, 0, size.y - 1);
				int k0 = glm::clamp(int(fmin(fkp, fmin(fkq, fkr))) - bandWidth, 0, size.z - 1);

				int i1 = glm::clamp(int(fmax(fip, fmax(fiq, fir))) + bandWidth + 1, 0, size.x - 1);
				int j1 = glm::clamp(int(fmax(fjp, fmax(fjq, fjr))) + bandWidth + 1, 0, size.y - 1);
				int k1 = glm::clamp(int(fmax(fkp, fmax(fkq, fkr))) + bandWidth + 1, 0, size.z - 1);


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
				j0 = glm::clamp((int)std::ceil(fmin(fjp, fmin(fjq, fjr))), 0, size.y - 1);
				k0 = glm::clamp((int)std::ceil(fmin(fkp, fmin(fkq, fkr))), 0, size.z - 1);

				j1 = glm::clamp((int)std::floor(fmax(fjp, fmax(fjq, fjr))), 0, size.y - 1);
				k1 = glm::clamp((int)std::floor(fmax(fkp, fmax(fkq, fkr))), 0, size.z - 1);

				for (int k = k0; k <= k1; k++) {
					for (int j = j0; j <= j1; j++) {
						float a, b, c;
						if (GetBarycentricCoordinates(j, k, fjp, fkp, fjq, fkq, fjr, fkr, &a, &b, &c)) {
							float fi = a * fip + b * fiq + c * fir;
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
		
		// TODO: convert into a kernel
		__device__ void PropagateDistanceField() {
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
			glm::ivec3 g, n, nbs[6];
			while (startIdx < (int)queue.size()) {
				g = queue[startIdx];
				startIdx++;

				GetNeighbourGridIndices6(g, nbs);
				for (int nIdx = 0; nIdx < 6; nIdx++) {
					n = nbs[nIdx];
					if (IsGridIndexInRange(n, size.x, size.y, size.z) && searchGrid(n) == false) {
						searchGrid.Set(n, true);
						queue.push_back(n);
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
					n = nbs[nIdx];
					if (IsGridIndexInRange(n, size.x, size.y, size.z) && ClosestTriangles(n) != -1) {
						t = MeshTriangles[ClosestTriangles(n)];
						float dist = PointToTriangleDistance(gPos, MeshVertices[t.x], MeshVertices[t.y], MeshVertices[t.z]);
						if (dist < Phi(g)) {
							Phi.Set(g, dist);
							ClosestTriangles.Set(g, ClosestTriangles(n));
						}
					}
				}
			}
		}

		// TODO: convert into a kernel
		__device__ __host__ void ComputeDistanceFieldSigns(Array3D<int>& intersectionCounts) {
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

		template <typename T>
		__device__ T Clamp(const T& n, const T& lower, const T& upper) {
			return max(lower, min(n, upper));
		}

		__device__ __host__ float PointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3) {
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
					float d1 = PointToSegmentDistance(x0, x1, x2);
					float d2 = PointToSegmentDistance(x0, x1, x3);
					return fmin(d1, d2);
				}
				else if (w31 > 0) {
					// this rules out edge 1-3
					float d1 = PointToSegmentDistance(x0, x1, x2);
					float d2 = PointToSegmentDistance(x0, x2, x3);
					return fmin(d1, d2);
				}
				else {
					// w12 must be >0, ruling out edge 1-2
					float d1 = PointToSegmentDistance(x0, x1, x3);
					float d2 = PointToSegmentDistance(x0, x2, x3);
					return fmin(d1, d2);
				}
			}
		}

		__device__ __host__ float PointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2) {
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

		__device__ __host__ bool GetBarycentricCoordinates(
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
			int signa = Orientation(x2, y2, x3, y3, &oa);
			if (signa == 0) {
				return false;
			}

			float ob;
			int signb = Orientation(x3, y3, x1, y1, &ob);
			if (signb != signa) {
				return false;
			}

			float oc;
			int signc = Orientation(x1, y1, x2, y2, &oc);
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

		__device__ __host__ int Orientation(	float x1, float y1, float x2, float y2, float* twiceSignedArea)	{

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

		__device__ void Negate() {
			for (int k = 0; k < Phi.Size.z; k++) {
				for (int j = 0; j < Phi.Size.y; j++) {
					for (int i = 0; i < Phi.Size.x; i++) {
						Phi.Set(i, j, k, -Phi(i, j, k));
					}
				}
			}
		}

		__device__ float operator()(int i, int j, int k) {
			return Get(i, j, k);
		}

		__device__ float Get(int i, int j, int k) {
			ASSERT(Phi.IsIndexInRange(i, j, k), "index out of range!");
			return Phi(i, j, k);
		}

		__device__ void CalculateUnion(MeshLevelSet& levelSet) {
			glm::ivec3 size = levelSet.Size;
			ASSERT(size == Size, "level set dimensions are not the same!");

			int indexOffset = MeshVertexCount;
			std::vector<glm::vec3> vertices(MeshVertices, MeshVertices + MeshVertexCount);
			vertices.insert(vertices.end(), levelSet.MeshVertices, levelSet.MeshVertices + levelSet.MeshVertexCount);

			glm::ivec3 t;
			std::vector<glm::ivec3> triangles(MeshTriangles, MeshTriangles + MeshTriangleCount);
			triangles.reserve(MeshTriangleCount + levelSet.MeshTriangleCount);
			for (size_t i = 0; i < levelSet.MeshTriangleCount; i++) {
				t = levelSet.MeshTriangles[i];
				t.x += indexOffset;
				t.y += indexOffset;
				t.z += indexOffset;
				triangles.push_back(t);
			}

			for (int k = 0; k < Phi.Size.z; k++) {
				for (int j = 0; j < Phi.Size.y; j++) {
					for (int i = 0; i < Phi.Size.x; i++) {
						if (levelSet.Get(i, j, k) < Phi(i, j, k)) {
							Phi.Set(i, j, k, levelSet.Get(i, j, k));

							int tidx = levelSet.GetClosestTriangleIndex(i, j, k) + indexOffset;
							ClosestTriangles.Set(i, j, k, tidx);
						}
					}
				}
			}

			MeshVertices = vertices.data();
			MeshVertexCount = vertices.size();

			MeshTriangles = triangles.data();
			MeshTriangleCount = triangles.size();
		}

		__device__ int GetClosestTriangleIndex(int i, int j, int k) {
			ASSERT(ClosestTriangles.IsIndexInRange(i, j, k), "index out of range!");
			return ClosestTriangles(i, j, k);
		}

		__device__ float TrilinearInterpolate(const glm::vec3& pos) {
			return Interpolation::TrilinearInterpolate(pos, DX, Phi);
		}

		__host__ void CalculateSDFN(const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount, int bandWidth = -1);

		// Mesh
		int MeshVertexCount;
		int MeshTriangleCount;
		const glm::vec3* MeshVertices;
		const glm::ivec3* MeshTriangles;

		float DX;

		glm::ivec3 Size;

		Array3D<float> Phi;
		Array3D<int> ClosestTriangles;
	};
}

#endif // !MESH_LEVEL_SET_CUH 