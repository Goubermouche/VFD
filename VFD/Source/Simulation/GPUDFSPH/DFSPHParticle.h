#ifndef DFSPH_PARTICLE_H
#define DFSPH_PARTICLE_H

#include "pch.h"

namespace vfd
{
	// 104 bytes
	struct DFSPHParticle
	{
		glm::vec3 Position;
		glm::vec3 Velocity;
		glm::vec3 Acceleration;

		float Mass;
		float Density;
		float Kappa;
		float KappaVelocity;

		// Viscosity
		glm::vec3 ViscosityDifference;

		// Surface Tension
		glm::vec3 MonteCarloSurfaceNormals;
		glm::vec3 MonteCarloSurfaceNormalsSmooth;

		float FinalCurvature;
		float DeltaFinalCurvature;
		float SmoothedCurvature;
		float MonteCarloSurfaceCurvature;
		float MonteCarloSurfaceCurvatureSmooth;
		float ClassifierInput;
		float ClassifierOutput;
	};

	// TEMP, DEBUG, only used when the scene gets reset
	struct DFSPHParticle0
	{
		glm::vec3 Position;
		glm::vec3 Velocity;
	};
}

#endif // !DFSPH_PARTICLE_H