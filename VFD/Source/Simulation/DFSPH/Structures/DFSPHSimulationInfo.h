#ifndef DFSPH_SIMULATION_INFO_H
#define DFSPH_SIMULATION_INFO_H

namespace vfd {
	struct DFSPHSimulationInfo {
		unsigned int ParticleCount = 0u;
		unsigned int RigidBodyCount = 0u;

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
		float ParticleMass;
		float ParticleMassInverse; // 1.0f / ParticleMassInverse

		// Viscosity
		float Viscosity;
		float BoundaryViscosity;
		float DynamicViscosity;
		float DynamicBoundaryViscosity;
		float TangentialDistanceFactor;
		float TangentialDistance;

		// Surface tension
		float SurfaceTension;
		unsigned int SurfaceTensionSampleCount;
		float ClassifierSlope;
		float ClassifierConstant;
		bool TemporalSmoothing;
		float SmoothingFactor;
		float Factor;
		float NeighborParticleRadius;
		float MonteCarloFactor;

		glm::vec3 Gravity;
	};
}

#endif // !DFSPH_SIMULATION_INFO_H