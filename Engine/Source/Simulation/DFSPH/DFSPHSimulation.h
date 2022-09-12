#ifndef DFSPH_SIMULATION_H
#define DFSPH_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Simulation/DFSPH/CompactNSearch.h"

// Inspired by: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH

namespace fe {

	struct DFSPHSimulationDescription {

	};

	// -------------------
	// Scene: 
	// -------------------
	// timeStepSize = 0.001;
	// numberOfStepsPerRenderUpdate = 2,
	// particleRadius = 0.025
	// density0 = 1000
	// simulationMethod = 4 
	// gravitation = {0, -9.81, 0}
	// cflMethod = 1
	// cflFactor = 1
	// cflMaxTimeStepSize = 0.005
	// maxIterations = 100
	// maxError = 0.1
	// maxIterationsV = 100
	// maxErrorV = 0.1
	// stiffness = 50000
	// exponent = 7
	// velocityUpdateMethod = 0
	// enableDivergenceSolver = true
	// boundaryHandlingMethod = 2
	// -------------------
	// Rigid bodies
	// -------------------
	// Box
	//   geometryFile = "../models/UnitBox.obj"
	//   translation = {0, -0.25, 0}
	//   rotationAxis = {1, 0, 0}
	//   rotationAngle = 0
	//   scale = {5, 0.5, 5}
	//   isDynamic = false
	//   isWall = false
	//   mapInvert = false
	//   mapThickness = 0.0
	//   mapResolution = {30, 20, 30}
	// Dragon
	//   geometryFile = "../models/Dragon_50k.obj"
	//   translation = {0, 0.5, 0}
	//   rotationAxis = {0, 1, 0}
	//   rotationAngle = 0
	//   scale = {2, 2, 2}
	//   isDynamic = false
	//   isWall = false
	//   mapInvert = false
	//   mapThickness = 0.0
	//   mapResolution = {20, 20, 20}
	// 
	// -------------------
	// Fluid models
	// -------------------
	// Bunny
	//   particleFile = "../models/bunny.bgeo"
	//   translation = {0.0, 1.8, -0.2}
	//   rotationAxis = {0, 1, 0}
	//   rotationAngle = 1.57

	/// <summary>
	/// SPH simulation wrapper
	/// </summary>
	class DFSPHSimulation : public RefCounted
	{
	public:
		DFSPHSimulation(const DFSPHSimulationDescription& desc);
		~DFSPHSimulation();

		void OnUpdate();
	public:
		bool paused = false;
	private:
		NeighborhoodSearch* m_NeighbourHoodSearch;
	};
}

#endif // !DFSPH_SIMULATION_H