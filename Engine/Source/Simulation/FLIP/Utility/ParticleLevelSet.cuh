#ifndef PARTICLE_LEVEL_SET_CUH
#define PARTICLE_LEVEL_SET_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/MeshLevelSet.cuh"

namespace fe {
	struct ParticleLevelSet {
		__device__ ParticleLevelSet() {}

		__device__ void Init(int i, int j, int k, double dx) {
			Size = { i, j, k };
			DX = dx;
			Phi.Init(i, j, k, GetMaxDistance());

			LOG("particle level set initialized", "FLIP", ConsoleColor::Cyan);
		}

		__device__ __host__ float GetMaxDistance() {
			return 3.0f * (float)DX;
		}

		__host__ __device__ void HostFree() {
			Phi.HostFree();
		}

		__device__ __host__ float operator()(int i, int j, int k) {
			return Get(i, j, k);
		}

		__device__ __host__ float operator()(glm::ivec3 g) {
			return Get(g);
		}

		__device__ __host__ float Get(glm::ivec3 g) {
			ASSERT(IsGridIndexInRange(g, Size.x, Size.y, Size.z), "index out of range!");
			return Phi(g);
		}

		__device__ __host__ float Get(int i, int j, int k) {
			ASSERT(IsGridIndexInRange({i, j, k}, Size.x, Size.y, Size.z), "index out of range!");
			return Phi(i, j, k);
		}

		glm::ivec3 Size;
		double DX;
		Array3D<float> Phi;
	};
}

#endif // !PARTICLE_LEVEL_SET_CUH