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
		float Density;
		float DensityAdvection;
		float PressureRho2;
		float PressureRho2V;
		float Factor;

		// Viscosity
		glm::vec3 ViscosityDifference;

		// Surface Tension
		glm::vec3 MonteCarloSurfaceNormal;
		glm::vec3 MonteCarloSurfaceNormalSmooth;

		float MonteCarloSurfaceCurvature;
		float MonteCarloSurfaceCurvatureSmooth;
		float DeltaFinalCurvature;
	};
}

#endif // !DFSPH_PARTICLE_H