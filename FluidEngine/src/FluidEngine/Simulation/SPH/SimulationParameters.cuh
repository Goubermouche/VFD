#ifndef SPH_SIMULATION_DESCRIPTION_CUH_
#define SPH_SIMULATION_DESCRIPTION_CUH_

#include "pch.h"

namespace fe {
	struct SimulationData 
	{
		float timeStep;
		float globalDamping;
		float particleRadius;
		float homogenity;
		float smoothingRadius;
		float spikyKern;
		float lapKern;
		float poly6Kern;
		float particleMass;
		float restDensity;
		float stiffness;
		float viscosity;
		float minDens;
		float minDist;
		float boundsHardDistance;
		float boundsSoftDistance;
		float boundsDamping;
		float boundsStiffness;
		float boundsDampingCritical;

		uint32_t particleCount;
		uint16_t maxParticlesInCellCount;
		uint32_t cellCount;

		uint32_t gridSizeYX;

		glm::vec3 gravity;
		glm::vec3 cellSize;
		glm::vec3 worldMin;
		glm::vec3 worldMax;
		glm::vec3 worldSize;
		glm::vec3 worldMinReal;
		glm::vec3 worldMaxReal;
		glm::vec3 worldSizeReal;

		glm::uvec3 gridSize;
	};
}

#endif // !SPH_SIMULATION_DESCRIPTION_CUH_