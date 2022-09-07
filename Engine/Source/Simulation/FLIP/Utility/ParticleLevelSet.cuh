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

		__host__ void CalculateSignedDistanceField(std::vector<glm::vec3>& particles, double radius, MeshLevelSet& solidPhi) {
			CalculateSDFFromParticles(particles, radius);
			ExtrapolateSDFIntoSolids(solidPhi);
		}

		__host__ void CalculateSDFFromParticles(std::vector<glm::vec3>& particles, double radius) {
			Phi.Fill(GetMaxDistance());

			glm::ivec3 g, gmin, gmax;
			glm::vec3 p;
			for (size_t pidx = 0; pidx < particles.size(); pidx++) {
				p = particles[pidx];
				g = PositionToGridIndex(particles[pidx], DX);
				gmin = glm::ivec3((int)fmax(0, g.x - 1), (int)fmax(0, g.y - 1), (int)fmax(0, g.z - 1));
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

		__host__ void ExtrapolateSDFIntoSolids(MeshLevelSet& solidPhi) {
			for (int k = 0; k < Size.z; k++) {
				for (int j = 0; j < Size.y; j++) {
					for (int i = 0; i < Size.x; i++) {
						if (Phi(i, j, k) < 0.5 * DX) {
							if (solidPhi.GetDistanceAtCellCenter(i, j, k) < 0) {
								Phi.Set(i, j, k, -0.5f * (float)DX);
							}
						}
					}
				}
			}
		}

		glm::ivec3 Size;
		double DX;
		Array3D<float> Phi;
	};
}

#endif // !PARTICLE_LEVEL_SET_CUH