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
		glm::vec3 PressureAcceleration;

		float PressureResiduum;
		float Mass;
		float Density;
		float DensityAdvection;
		float PressureRho2;
		float PressureRho2V;
		float Factor;

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
}

#endif // !DFSPH_PARTICLE_H