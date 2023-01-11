#ifndef GPU_DFSPH_SIMULATION_DESCRIPTION_H
#define GPU_DFSPH_SIMULATION_DESCRIPTION_H

#include "Simulation/GPUDFSPH/RigidBody/RigidBody.cuh"

namespace vfd
{
	struct DFSPHSimulationDescription
	{
		// Time step
		float TimeStepSize = 0.001f;
		float MinTimeStepSize = 0.0001f;
		float MaxTimeStepSize = 0.005f;

		// Pressure solver
		unsigned int MinPressureSolverIterations = 2;
		unsigned int MaxPressureSolverIterations = 100;
		float MaxPressureSolverError = 10.0f; // Highest allowed pressure solver error [%]

		// Divergence solver
		bool EnableDivergenceSolverError = true;
		unsigned int MinDivergenceSolverIterations = 0;
		unsigned int MaxDivergenceSolverIterations = 100;
		float MaxDivergenceSolverError = 10.0f; // Highest allowed divergence solver error [%]

		// Viscosity solver
		unsigned int MinViscositySolverIterations = 0;
		unsigned int MaxViscositySolverIterations = 100;
		float MaxViscositySolverError = 1.0f; // Highest allowed viscosity solver error [%]
		float Viscosity = 1.0f;
		float BoundaryViscosity = 1.0f;
		float TangentialDistanceFactor = 0.3f;

		// Scene
		float ParticleRadius = 0.025f;
		glm::vec3 Gravity = { 0.0f, -9.81f, 0.0f };

		std::vector<RigidBodyDescription> BoundaryObjects;
	};

	inline bool operator==(const DFSPHSimulationDescription& lhs, const DFSPHSimulationDescription& rhs)
	{
		return
			lhs.TimeStepSize == rhs.TimeStepSize &&
			lhs.MinTimeStepSize == rhs.MinTimeStepSize &&
			lhs.MaxTimeStepSize == rhs.MaxTimeStepSize &&
			lhs.MinPressureSolverIterations == rhs.MinPressureSolverIterations &&
			lhs.MaxPressureSolverIterations == rhs.MaxPressureSolverIterations &&
			lhs.MaxPressureSolverError == rhs.MaxPressureSolverError &&
			lhs.EnableDivergenceSolverError == rhs.EnableDivergenceSolverError &&
			lhs.MinDivergenceSolverIterations == rhs.MinDivergenceSolverIterations &&
			lhs.MaxDivergenceSolverIterations == rhs.MaxDivergenceSolverIterations &&
			lhs.MaxDivergenceSolverError == rhs.MaxDivergenceSolverError &&
			lhs.MinViscositySolverIterations == rhs.MinViscositySolverIterations &&
			lhs.MaxViscositySolverIterations == rhs.MaxViscositySolverIterations &&
			lhs.MaxViscositySolverError == rhs.MaxViscositySolverError &&
			lhs.Viscosity == rhs.Viscosity &&
			lhs.BoundaryViscosity == rhs.BoundaryViscosity &&
			lhs.TangentialDistanceFactor == rhs.TangentialDistanceFactor &&
			lhs.ParticleRadius == rhs.ParticleRadius &&
			lhs.Gravity == rhs.Gravity;
	}
}

#endif // !GPU_DFSPH_SIMULATION_DESCRIPTION_H