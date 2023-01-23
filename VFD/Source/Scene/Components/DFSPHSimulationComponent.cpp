#include "pch.h"
#include "DFSPHSimulationComponent.h"

namespace vfd {
	DFSPHSimulationComponent::DFSPHSimulationComponent()
	{
		DFSPHSimulationDescription description;

		// Time step
		description.TimeStepSize = 0.001f;
		description.MinTimeStepSize = 0.0001f;
		description.MaxTimeStepSize = 0.005f;

		description.FrameLength = 0.016f;
		description.FrameCount = 200u;

		// Pressure solver
		description.MinPressureSolverIterations = 0u;
		description.MaxPressureSolverIterations = 100u;
		description.MaxPressureSolverError = 10.0f;

		// Divergence solver
		description.EnableDivergenceSolverError = true;
		description.MinDivergenceSolverIterations = 0u;
		description.MaxDivergenceSolverIterations = 100u;
		description.MaxDivergenceSolverError = 10.0f;

		// Viscosity solver
		description.MinViscositySolverIterations = 0u;
		description.MaxViscositySolverIterations = 100u;
		description.MaxViscositySolverError = 0.1f;
		description.Viscosity = 10.0f;
		description.BoundaryViscosity = 10.0f;
		description.TangentialDistanceFactor = 0.5f;

		// Surface tension
		description.EnableSurfaceTensionSolver = false;

		// Scene
		description.ParticleRadius = 0.025f;
		description.Gravity = { 0.0f, -9.81f, 0.0f };

		Handle = Ref<DFSPHSimulation>::Create(description);
	}

	DFSPHSimulationComponent::DFSPHSimulationComponent(DFSPHSimulationDescription& description)
		: Handle(Ref<DFSPHSimulation>::Create(description))
	{}

	DFSPHSimulationComponent::DFSPHSimulationComponent(Ref<DFSPHSimulation> simulation)
		: Handle(simulation)
	{}
}
