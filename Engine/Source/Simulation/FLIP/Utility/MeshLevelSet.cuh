#ifndef MESH_LEVEL_SET_CUH 
#define MESH_LEVEL_SET_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Compute/Utility/CUDA/cutil_math.h"
#include "Simulation/FLIP/Utility/Interpolation.cuh"
#include "Compute/Utility/CudaKernelUtility.cuh"
#include "Renderer/Mesh/TriangleMesh.h"
#include "Simulation/FLIP/Utility/LevelSetUtils.cuh"

namespace fe {

	static __global__ void CalculateExactBandDistanceFieldKernel(int bandWidth, float DX, float invDX, glm::ivec3 size, const glm::vec3* vertices, int vertexCount, const glm::ivec3* triangles, int triangleCount);
	static __global__ void CalculateDistanceFieldSignsKernel(int sizeX);

	struct MeshLevelSet {
		__host__ void Init(int i, int j, int k, float dx) {
			Size = { i, j, k };
			DX = dx;
			Phi.Init(i + 1, j + 1, k + 1, 0.0f);
			ClosestTriangles.Init(i + 1, j + 1, k + 1, -1);
		}

		__host__ void Init(TriangleMesh& mesh, int resolution, float dx, int bandWidth = -1);

		__host__ void Negate() {
			for (int k = 0; k < Phi.Size.z; k++) {
				for (int j = 0; j < Phi.Size.y; j++) {
					for (int i = 0; i < Phi.Size.x; i++) {
						Phi.Set(i, j, k, -Phi(i, j, k));
					}
				}
			}
		}

		__host__ float operator()(int i, int j, int k) {
			return Get(i, j, k);
		}

		__host__ float Get(int i, int j, int k) {
			ASSERT(Phi.IsIndexInRange(i, j, k), "index out of range!");
			return Phi(i, j, k);
		}

		__host__ void CalculateUnion(MeshLevelSet& other) {
			ASSERT(Size == other.Size, "level set dimensions are not the same!");

			for (int k = 0; k < Phi.Size.z; k++) {
				for (int j = 0; j < Phi.Size.y; j++) {
					for (int i = 0; i < Phi.Size.x; i++) {
						if (other(i, j, k) < Phi(i, j, k)) {
							Phi.Set(i, j, k, other(i, j, k));

							int tidx = other.GetClosestTriangleIndex(i, j, k) + MeshVertexCount;
							ClosestTriangles.Set(i, j, k, tidx);
						}
					}
				}
			}

			// vertices
			glm::vec3* vertices = new glm::vec3[MeshVertexCount + other.MeshVertexCount];
			std::copy(MeshVertices, MeshVertices + MeshVertexCount, vertices);
			std::copy(other.MeshVertices, other.MeshVertices + other.MeshVertexCount, vertices + MeshVertexCount);
			delete[] MeshVertices;
			MeshVertices = vertices;

			// triangles
			glm::ivec3* triangles = new glm::ivec3[MeshTriangleCount + other.MeshTriangleCount];
			std::copy(MeshTriangles, MeshTriangles + MeshTriangleCount, triangles);

			glm::ivec3 t;
			int triangleIndex = MeshVertexCount;
			for (size_t i = 0; i < other.MeshTriangleCount; i++)
			{
				t = other.MeshTriangles[i];
				t.x += MeshVertexCount;
				t.y += MeshVertexCount;
				t.z += MeshVertexCount;
				triangles[triangleIndex] = t;
				triangleIndex++;
			}

			delete[] MeshTriangles;
			MeshTriangles = triangles;
		}

		__host__ int GetClosestTriangleIndex(int i, int j, int k) {
			ASSERT(ClosestTriangles.IsIndexInRange(i, j, k), "index out of range!");
			return ClosestTriangles(i, j, k);
		}

		__host__ float TrilinearInterpolate(const glm::vec3& pos) {
			return Interpolation::TrilinearInterpolate(pos, DX, Phi);
		}

		__host__ glm::vec3 TrilinearInterpolateGradient(glm::vec3 pos) {
			glm::vec3 grad;
			Interpolation::TrilinearInterpolateGradient(pos, DX, Phi, &grad);
			return grad;
		}

		__host__ float GetDistanceAtCellCenter(int i, int j, int k) {
			ASSERT(IsGridIndexInRange({ i, j, k }, Size.x, Size.y, Size.z), "index out of range!");
			return 0.125f * (Phi(i, j, k) +
				Phi(i + 1, j, k) +
				Phi(i, j + 1, k) +
				Phi(i + 1, j + 1, k) +
				Phi(i, j, k + 1) +
				Phi(i + 1, j, k + 1) +
				Phi(i, j + 1, k + 1) +
				Phi(i + 1, j + 1, k + 1));
		}

		__host__ float GetFaceWeightU(int i, int j, int k) {
			// ASSERT(IsGridIndexInRange(i, j, k, _isize + 1, _jsize, _ksize));
			return LevelSetUtils::FractionInside(Phi(i, j, k),
				Phi(i, j + 1, k),
				Phi(i, j, k + 1),
				Phi(i, j + 1, k + 1));
		}

		__host__ float GetFaceWeightU(glm::ivec3 g) {
			return GetFaceWeightU(g.x, g.y, g.z);
		}

		__host__ float GetFaceWeightV(int i, int j, int k) {
			// ASSERT(Grid3d::isGridIndexInRange(i, j, k, _isize, _jsize + 1, _ksize));
			return LevelSetUtils::FractionInside(Phi(i, j, k),
				Phi(i, j, k + 1),
				Phi(i + 1, j, k),
				Phi(i + 1, j, k + 1));
		}

		__host__ float GetFaceWeightV(glm::ivec3 g) {
			return GetFaceWeightV(g.x, g.y, g.z);
		}

		__host__ float GetFaceWeightW(int i, int j, int k) {
			// ASSERT(Grid3d::isGridIndexInRange(i, j, k, _isize, _jsize, _ksize + 1));
			return LevelSetUtils::FractionInside(Phi(i, j, k),
				Phi(i, j + 1, k),
				Phi(i + 1, j, k),
				Phi(i + 1, j + 1, k));
		}

		float GetFaceWeightW(glm::ivec3 g) {
			return GetFaceWeightW(g.x, g.y, g.z);
		}

		__host__ __device__ void DeviceFree() {
			Phi.DeviceFree();
			ClosestTriangles.DeviceFree();

			if (MeshVertexCount > 0) {
				delete[] MeshVertices;
			}

			if (MeshTriangleCount > 0) {
				delete[] MeshTriangles;
			}
		}

		__host__ void HostFree() {
			Phi.HostFree();
			ClosestTriangles.HostFree();

			if (MeshVertexCount > 0) {
				delete[] MeshVertices;
			}

			if (MeshTriangleCount > 0) {
				delete[] MeshTriangles;
			}
		}

		// Mesh
		int MeshVertexCount;
		int MeshTriangleCount;

		glm::vec3* MeshVertices;
		glm::ivec3* MeshTriangles;

		float DX;

		glm::ivec3 Size;

		Array3D<float> Phi;
		Array3D<int> ClosestTriangles;
	};
}

#endif // !MESH_LEVEL_SET_CUH 