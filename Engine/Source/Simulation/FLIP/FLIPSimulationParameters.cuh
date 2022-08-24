#ifndef FLIP_SIMULATION_DATA_CUH
#define FLIP_SIMULATION_DATA_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/MarkerAndCellVelocityField.cuh"

namespace fe {
	struct FLIPSimulationParameters
	{ 
		float TimeStep; // Time step / Sub step count
		float DX;
		float ParticleRadius;

		uint32_t SubStepCount;

		glm::ivec3 Size; // Grid size

	};
}

#endif // !FLIP_SIMULATION_DATA_CUH