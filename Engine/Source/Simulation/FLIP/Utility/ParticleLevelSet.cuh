#ifndef PARTICLE_LEVEL_SET_CUH
#define PARTICLE_LEVEL_SET_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"

namespace fe {
	struct ParticleLevelSet {
		__device__ ParticleLevelSet() {}

		__device__ void Init(int i, int j, int k, float dx) {
			Size = { i, j, k };
			DX = dx;
			Phi.Init(i, j, k, GetMaxDistance());

			LOG("particle level set initialized", "FLIP", ConsoleColor::Cyan);
		}

		__device__ float GetMaxDistance() {
			return 3.0f * DX;
		}

		__host__ __device__ void HostFree() {
			Phi.HostFree();
		}

		glm::ivec3 Size;
		float DX;
		Array3D<float> Phi;
	};
}

#endif // !PARTICLE_LEVEL_SET_CUH