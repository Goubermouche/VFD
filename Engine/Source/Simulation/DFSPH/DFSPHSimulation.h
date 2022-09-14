#ifndef DFSPH_SIMULATION_H
#define DFSPH_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Simulation/DFSPH/CompactNSearch.h"
#include "Simulation/DFSPH/StaticBoundarySimulator.h"

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

	struct BoundaryData {
		std::string samplesFile;
		std::string meshFile;
		glm::vec3 translation;
		glm::mat4 rotation;
		glm::vec3 scale;
		float density;
		bool dynamic;
		bool isWall;
		void* rigidBody;

		std::string mapFile;
		bool mapInvert;
		float mapThickness;
		glm::ivec3 mapResolution;
		unsigned int samplingMode;
		bool isAnimated;
	};

	struct FluidData {
		std::string id;
		std::string samplesFile;
		std::string visMeshFile;
		glm::vec3 translation;
		glm::mat4 rotation;
		glm::vec3 scale;
		glm::vec3 initialVelocity;
		glm::vec3 initialAngularVelocity;
		unsigned char mode;
		bool invert;
		glm::ivec3 resolutionSDF;
	};

	struct FluidBlock {

	};

	struct EmitterData {

	};

	struct AnimationFieldData {

	};

	struct MaterialData {
		std::string id;
		std::string colorField;
		unsigned int colorMapType;
		float minVal;
		float maxVal;
		unsigned int maxEmitterParticles;
		bool emitterReuseParticles;
		glm::vec3 emitterBoxMin;
		glm::vec3 emitterBoxMax;
	};

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
		unsigned int m_numberOfStepsPerRenderUpdate;
		std::string m_exePath;
		std::string m_stateFile;
		std::string m_outputPath;
		unsigned int m_currentObjectId;
		bool m_useParticleCaching;
		bool m_useGUI;
		bool m_isStaticScene;
		int m_renderWalls;
		bool m_doPause;
		float m_pauseAt;
		float m_stopAt;
		bool m_cmdLineStopAt;
		bool m_cmdLineNoInitialPause;
		bool m_enableRigidBodyVTKExport;
		bool m_enableRigidBodyExport;
		bool m_enableStateExport;
		bool m_enableAsyncExport;
		bool m_enableObjectSplitting;
		float m_framesPerSecond;
		float m_framesPerSecondState;
		std::string m_particleAttributes;
		float m_nextFrameTime;
		float m_nextFrameTimeState;
		bool m_firstState;
		unsigned int m_frameCounter;
		bool m_isFirstFrame;
		bool m_isFirstFrameVTK;
		std::vector<std::string> m_colorField;
		std::vector<int> m_colorMapType;
		std::vector<float> m_renderMaxValue;
		std::vector<float> m_renderMinValue;
		float const* m_colorMapBuffer;
		unsigned int m_colorMapLength;
		StaticBoundarySimulator* m_boundarySimulator;
		int m_argc;
		std::vector<char*> m_argv_vec;
		char** m_argv;
		std::string m_windowName;
		std::vector<std::string> m_paramTokens;
		std::function<void()> m_timeStepCB;
		std::function<void()> m_resetCB;
		std::vector<std::vector<float>> m_scalarField;
		bool m_updateGUI;

		// Scene 
		std::vector<BoundaryData*> boundaryModels;
		std::vector<FluidData*> fluidModels;
		std::vector<FluidBlock*> fluidBlocks;
		std::vector<EmitterData*> emitters;
		std::vector<AnimationFieldData*> animatedFields;
		std::vector<MaterialData*> materials;
		float particleRadius;
		float timeStepSize;
	};
}

#endif // !DFSPH_SIMULATION_H