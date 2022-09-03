#ifndef PARTICLE_LEVEL_SET_CUH
#define PARTICLE_LEVEL_SET_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"
#include "Simulation/FLIP/Utility/MeshLevelSet.cuh"

namespace fe {
	struct ParticleLevelSet {
		__device__ ParticleLevelSet() {}

		__device__ void Init(int i, int j, int k, float dx) {
			Size = { i, j, k };
			DX = dx;
			Phi.Init(i, j, k, GetMaxDistance());

			LOG("particle level set initialized", "FLIP", ConsoleColor::Cyan);
		}

		__device__ __host__ float GetMaxDistance() {
			return 3.0f * DX;
		}

		__host__ __device__ void HostFree() {
			Phi.HostFree();
		}

		__host__ void CalculateSDF(std::vector<glm::vec3>& particles, float radius, MeshLevelSet& solidPhi) {
			glm::ivec3 solidSize = solidPhi.Size;
			ASSERT(solidSize == Size, "SDF's have to be the same size!");

			CalculateSDFFromParticles(particles, radius);
			ExtrapolateSignedDistanceIntoSolids(solidPhi);
		}

		// TODO: convert to a kernel
		__host__ void CalculateSDFFromParticles(std::vector<glm::vec3>& particles, float radius) {
			Phi.Fill(GetMaxDistance());

			glm::ivec3 g, gmin, gmax;
			glm::vec3 p;

			for (size_t pidx = 0; pidx < particles.size(); pidx++)
			{
				p = particles[pidx];
				g = PositionToGridIndex(particles[pidx], DX);
				gmin = glm::ivec3 { ((int)fmax(0, g.x - 1), (int)fmax(0, g.y- 1), (int)fmax(0, g.z - 1)) };

				gmax = glm::ivec3((int)fmin(g.x + 1, Size.x - 1),
					(int)fmin(g.y + 1, Size.y - 1),
					(int)fmin(g.z + 1, Size.z - 1));

				for (int k = gmin.z; k <= gmax.z; k++) {
					for (int j = gmin.y; j <= gmax.y; j++) {
						for (int i = gmin.x; i <= gmax.x; i++) {
							glm::vec3 cpos = GridIndexToCellCenter(i, j, k, DX);
							float dist = glm::length(cpos - p) - (float)radius;
							if (dist < Phi(i, j, k)) {
								Phi.Set(i, j, k, dist);
							}
						}
					}
				}
			}
		}

		// TODO: convert to a kernel
		__host__ void ExtrapolateSignedDistanceIntoSolids(MeshLevelSet& solidPhi) {
			for (int k = 0; k < Size.z; k++) {
				for (int j = 0; j < Size.y; j++) {
					for (int i = 0; i < Size.x; i++) {
						if (Phi(i, j, k) < 0.5 * DX) {
							if (solidPhi.GetDistanceAtCellCenter(i, j, k) < 0) {
								Phi.Set(i, j, k, -0.5f * DX);
							}
						}
					}
				}
			}
		}

		glm::ivec3 Size;
		float DX;
		Array3D<float> Phi;
	};
}

#endif // !PARTICLE_LEVEL_SET_CUH