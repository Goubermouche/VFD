#ifndef DFSPH_SIMULATION_INFO_H
#define DFSPH_SIMULATION_INFO_H

namespace vfd {
	struct DFSPHSimulationInfo {
		unsigned int ParticleCount;
		unsigned int RigidBodyCount;

		float SupportRadius;
		float SupportRadius2;
		float ParticleRadius;
		float ParticleDiameter;

		float TimeStepSize;
		float TimeStepSize2;
		float TimeStepSizeInverse;
		float TimeStepSize2Inverse;

		float Volume;
		float Density0;

		float Viscosity;
		float BoundaryViscosity;
		float DynamicViscosity;
		float DynamicBoundaryViscosity;
		float TangentialDistanceFactor;
		float TangentialDistance;

		glm::vec3 Gravity;
	};
}

#endif // !DFSPH_SIMULATION_INFO_H