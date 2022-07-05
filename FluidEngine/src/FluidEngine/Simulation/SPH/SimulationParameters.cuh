#ifndef SPH_SIMULATION_PARAMETERS_CUH_
#define SPH_SIMULATION_PARAMETERS_CUH_

namespace fe {
	struct SimulationParameters
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

		unsigned int particleCount;
		unsigned int maxParticlesInCellCount;
		unsigned int gridSizeYX;
		unsigned int cellCount;

		float3 gravity;
		float3 cellSize;
		float3 worldMin;
		float3 worldMax;
		float3 worldSize;
		float3 worldMinReal;
		float3 worldMaxReal;
		float3 worldSizeReal;

		uint3 gridSize;
	};
}

#endif // !SPH_SIMULATION_PARAMETERS_CUH_