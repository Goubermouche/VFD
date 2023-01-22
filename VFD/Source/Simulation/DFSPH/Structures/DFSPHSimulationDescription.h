#ifndef GPU_DFSPH_SIMULATION_DESCRIPTION_H
#define GPU_DFSPH_SIMULATION_DESCRIPTION_H

#include "Simulation/DFSPH/RigidBody/RigidBody.cuh"
#include "Simulation/DFSPH/FluidObject/FluidObject.h"

namespace vfd
{
	struct DFSPHSimulationDescription
	{
		// Time step
		float TimeStepSize = 0.001f;
		float MinTimeStepSize = 0.0001f;
		float MaxTimeStepSize = 0.005f;

		float FrameLength = 0.0016f;
		unsigned int FrameCount = 1000u;

		// Pressure solver
		unsigned int MinPressureSolverIterations = 0u;
		unsigned int MaxPressureSolverIterations = 100u;
		float MaxPressureSolverError = 10.0f; // Highest allowed pressure solver error [%]

		// Divergence solver
		bool EnableDivergenceSolverError = true;
		unsigned int MinDivergenceSolverIterations = 0u;
		unsigned int MaxDivergenceSolverIterations = 100u;
		float MaxDivergenceSolverError = 10.0f; // Highest allowed divergence solver error [%]

		// Viscosity solver
		bool EnableViscositySolver = true;
		unsigned int MinViscositySolverIterations = 0u;
		unsigned int MaxViscositySolverIterations = 100u;
		float MaxViscositySolverError = 1.0f; // Highest allowed viscosity solver error [%]
		float Viscosity = 1.0f;
		float BoundaryViscosity = 1.0f;
		float TangentialDistanceFactor = 0.3f;

		// Surface solver
		bool EnableSurfaceTensionSolver = true;
		unsigned int SurfaceTensionSmoothPassCount = 1u;
		float SurfaceTension = 1.0f;
		bool TemporalSmoothing = false;
		int CSDFix = -1.0f; // Sample count per computational step
		int CSD = 10000; // Sample count per particle per second

		// Scene
		float ParticleRadius = 0.025f;
		glm::vec3 Gravity = { 0.0f, -9.81f, 0.0f };
	};

	inline bool operator==(const DFSPHSimulationDescription& lhs, const DFSPHSimulationDescription& rhs)
	{
		return
			lhs.TimeStepSize == rhs.TimeStepSize &&
			lhs.MinTimeStepSize == rhs.MinTimeStepSize &&
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
			lhs.Gravity == rhs.Gravity &&
			lhs.SurfaceTension == rhs.SurfaceTension &&
			lhs.SurfaceTensionSmoothPassCount == rhs.SurfaceTensionSmoothPassCount &&
			lhs.CSDFix == rhs.CSDFix &&
			lhs.CSD == rhs.CSD &&
			lhs.EnableSurfaceTensionSolver == rhs.EnableSurfaceTensionSolver &&
			lhs.EnableViscositySolver == rhs.EnableViscositySolver &&
			lhs.FrameLength == rhs.FrameLength &&
			lhs.FrameCount == rhs.FrameCount &&
			lhs.TemporalSmoothing == rhs.TemporalSmoothing;
	}
}

#endif // !GPU_DFSPH_SIMULATION_DESCRIPTION_H