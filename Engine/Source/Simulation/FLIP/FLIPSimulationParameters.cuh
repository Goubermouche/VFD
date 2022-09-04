#ifndef FLIP_SIMULATION_DATA_CUH
#define FLIP_SIMULATION_DATA_CUH

#include "pch.h"
#include "Simulation/FLIP/Utility/MarkerAndCellVelocityField.cuh"
#include "Simulation/FLIP/Utility/ParticleLevelSet.cuh"
#include "Simulation/FLIP/Utility/PressureSolver.cuh"
#include "Simulation/FLIP/Utility/MeshLevelSet.cuh"
#include "Simulation/FLIP/Utility/Grid3D.cuh"
#include "Simulation/FLIP/Utility/ViscositySolver.cuh"
#include "Simulation/FLIP/Utility/LevelSetUtils.cuh"

namespace fe {
	struct FLIPSimulationParameters
	{ 
		int Resolution;
		int ParticleCount;

		float DX;
		float ParticleRadius;
		float Viscosity;

		uint32_t SubStepCount;

		glm::vec3 Gravity;
	};
}

#endif // !FLIP_SIMULATION_DATA_CUH