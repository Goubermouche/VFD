#ifndef GPU_DFSPH_SIMULATION_DESCRIPTION_H
#define GPU_DFSPH_SIMULATION_DESCRIPTION_H

namespace vfd
{
	struct GPUDFSPHSimulationDescription
	{
		// Time step
		float TimeStepSize = 0.001f;
		float MinTimeStepSize = 0.0001f;
		float MaxTimeStepSize = 0.005f;

		// Pressure solver
		unsigned int MinPressureSolverIterations = 2;
		unsigned int MaxPressureSolverIterations = 100;
		float MaxPressureSolverError = 0.1f; // Highest allowed pressure solver error [%]

		// Divergence solver
		bool EnableDivergenceSolverError = true;
		unsigned int MinDivergenceSolverIterations = 0;
		unsigned int MaxDivergenceSolverIterations = 100;
		float MaxDivergenceSolverError = 0.1f; // Highest allowed divergence solver error [%]

		// Viscosity solver
		unsigned int MinViscositySolverIterations = 0;
		unsigned int MaxViscositySolverIterations = 100;
		float MaxViscositySolverError = 0.001f; // Highest allowed viscosity solver error [%]

		// Scene
		float ParticleRadius = 0.025f;
		glm::vec3 Gravity = { 0.0f, -9.81f, 0.0f };
	};
}

#endif // !GPU_DFSPH_SIMULATION_DESCRIPTION_H