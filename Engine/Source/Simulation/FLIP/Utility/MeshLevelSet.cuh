#ifndef MESH_LEVEL_SET_CUH 
#define MESH_LEVEL_SET_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Compute/Utility/CUDA/cutil_math.h"
#include "Simulation/FLIP/Utility/Interpolation.cuh"
#include "Compute/Utility/CudaKernelUtility.cuh"

namespace fe {
	struct MeshLevelSet {
		__host__ void Init(int i, int j, int k, float dx) {
			Size = { i, j, k };
			DX = dx;
			Phi.Init(i + 1, j + 1, k + 1, 0.0f);
			ClosestTriangles.Init(i + 1, j + 1, k + 1, -1);
		}

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

			// vertices
			glm::vec3* vertices = new glm::vec3[MeshVertexCount + other.MeshVertexCount];
			std::copy(MeshVertices, MeshVertices + MeshVertexCount, vertices);
			std::copy(other.MeshVertices, other.MeshVertices + other.MeshVertexCount, vertices + MeshVertexCount);

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
				triangles[MeshVertexCount] = t;
				MeshVertexCount++;
			}

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

			// delete[] MeshVertices;
			// delete[] MeshTriangles;

			MeshVertexCount = MeshVertexCount + other.MeshVertexCount;
			MeshTriangleCount = MeshTriangleCount + other.MeshTriangleCount;

			MeshVertices = new glm::vec3[MeshVertexCount];
			MeshTriangles = new glm::ivec3[MeshTriangleCount];

			std::copy(vertices, vertices + MeshVertexCount, MeshVertices);
			std::copy(triangles, triangles + MeshTriangleCount, MeshTriangles);

		 	delete[] vertices;
			delete[] triangles;
		}

		__host__ int GetClosestTriangleIndex(int i, int j, int k) {
			ASSERT(ClosestTriangles.IsIndexInRange(i, j, k), "index out of range!");
			return ClosestTriangles(i, j, k);
		}

		__host__ float TrilinearInterpolate(const glm::vec3& pos) {
			return Interpolation::TrilinearInterpolate(pos, DX, Phi);
		}

		__host__ void CalculateSDF(glm::vec3* vertices, int vertexCount, glm::ivec3* triangles, int triangleCount, int bandWidth = -1);

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