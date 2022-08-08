#ifndef SPH_SIMULATION_DESCRIPTION_CUH
#define SPH_SIMULATION_DESCRIPTION_CUH

#include "pch.h"

namespace fe {
	struct SimulationData
	{
		float Time;
		float TimeStep;
		float GlobalDamping;
		float ParticleRadius;
		float Homogeneity;
		float SmoothingRadius;
		float SpikyKern;
		float LapKern;
		float Poly6Kern;
		float ParticleMass;
		float RestDensity;
		float Stiffness;
		float Viscosity;
		float MinDens;
		float MinDist;
		float BoundsHardDistance;
		float BoundsSoftDistance;
		float BoundsDamping;
		float BoundsStiffness;
		float BoundsDampingCritical;

		uint32_t ParticleCount;
		uint32_t GridSizeYX;
		uint32_t CellCount;

		uint16_t MaxParticlesInCellCount;

		glm::vec3 Gravity;
		glm::vec3 CellSize;
		glm::vec3 WorldMin;
		glm::vec3 WorldMax;
		glm::vec3 WorldSize;
		glm::vec3 WorldMinReal;
		glm::vec3 WorldMaxReal;
		glm::vec3 WorldSizeReal;

		glm::uvec3 GridSize;
	};
}

#endif // !SPH_SIMULATION_DESCRIPTION_CUH