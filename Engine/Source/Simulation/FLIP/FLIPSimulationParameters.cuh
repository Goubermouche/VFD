#ifndef FLIP_SIMULATION_DATA_CUH
#define FLIP_SIMULATION_DATA_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/MarkerAndCellVelocityField.cuh"
#include "Simulation/FLIP/Utility/ParticleLevelSet.cuh"
#include "Simulation/FLIP/Utility/PressureSolver.cuh"
#include "Simulation/FLIP/Utility/Mesh.cuh" // --
#include "Simulation/FLIP/Utility/MeshLevelSet.cuh"

namespace fe {
	struct FLIPSimulationParameters
	{ 
		float TimeStep; // Time step / Sub step count
		float DX;
		float ParticleRadius;

		uint32_t SubStepCount;

		glm::ivec3 Size; // Grid size
		glm::vec3 Gravity;

	};
}

#endif // !FLIP_SIMULATION_DATA_CUH