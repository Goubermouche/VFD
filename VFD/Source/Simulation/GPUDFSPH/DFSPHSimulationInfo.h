#ifndef DFSPH_SIMULATION_INFO_H
#define DFSPH_SIMULATION_INFO_H

namespace vfd {
	struct DFSPHSimulationInfo {
		unsigned int ParticleCount;
		unsigned int RigidBodyCount;

		float SupportRadius;
		float ParticleRadius;
		float ParticleDiameter;

		float TimeStepSize;
		float TimeStepSize2;
		float TimeStepSizeInverse;
		float TimeStepSize2Inverse;

		float Volume;
		float Density0;

		glm::vec3 Gravity;
	};
}

#endif // !DFSPH_SIMULATION_INFO_H