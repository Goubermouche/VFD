#ifndef FLIP_SIMULATION_DATA_CUH
#define FLIP_SIMULATION_DATA_CUH

#include "pch.h"

namespace fe {
	struct FLIPSimulationData
	{ 
		float TimeStep; // Time step / Sub step count
		float DX;

		uint32_t SubStepCount;

		glm::ivec3 Size; // Grid size

		__device__ void Test() {
			// std::printf("%.6f", TimeStep);
		}
	};

	struct FluidParticle {
		glm::vec3 Position;
		glm::vec3 Velocity;
	};
}

#endif // !FLIP_SIMULATION_DATA_CUH