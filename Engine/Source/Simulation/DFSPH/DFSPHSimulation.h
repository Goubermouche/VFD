#ifndef DFSPH_SIMULATION_H
#define DFSPH_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Simulation/DFSPH/CompactNSearch.h"
#include "Simulation/DFSPH/StaticBoundarySimulator.h"
#include "Kernel.h"

// Inspired by: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH

namespace fe {
	class SurfaceTensionZorillaRitter2020 {

	};

	class ViscosityWeiler2018 {

	};

	struct DFSPHSimulationDescription {

	};

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

	class DFSPHSimulation;
	class SimulationDataDFSPH {
	public:
		SimulationDataDFSPH();
		void Init(DFSPHSimulation* sim);
	protected:
		std::vector<std::vector<float>> m_factor;
		std::vector<std::vector<float>> m_kappa;
		std::vector<std::vector<float>> m_kappaV;
		std::vector<std::vector<float>> m_density_adv;

	};

	class StaticBoundarySimulator;

	typedef PrecomputedKernel<CubicKernel, 10000> PrecomputedCubicKernel;

	/// <summary>
	/// SPH simulation wrapper
	/// </summary>
	class DFSPHSimulation : public RefCounted
	{
	public:
		DFSPHSimulation(const DFSPHSimulationDescription& desc);
		~DFSPHSimulation();

		void OnUpdate();
		void OnRenderTemp();
	private:
		void SetParticleRadius(float val);
		void BuildModel();

		void InitFluidData();
	public:
		bool paused = false;
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
		std::vector<MaterialData*> materials;
		float particleRadius;
		float timeStepSize;

		// sim 
		float m_supportRadius;
		
		bool m_enableZSort;
		glm::vec3 m_gravitation;
		float m_cflFactor;
		int m_cflMethod;
		float m_cflMinTimeStepSize;
		float m_cflMaxTimeStepSize;
		int m_boundaryHandlingMethod;
		NeighborhoodSearch* m_neighborhoodSearch;

		// fluid model
		enum class ParticleState { Active = 0, AnimatedByEmitter, Fixed };

		int m_numParticles;
		unsigned int m_numActiveParticles;
		unsigned int m_numActiveParticles0;

		std::vector<glm::vec3> m_x;
		std::vector<glm::vec3> m_x0;
		std::vector<glm::vec3> m_v;
		std::vector<glm::vec3> m_v0;
		std::vector<glm::vec3> m_a;
		std::vector<float> m_masses;
		std::vector<float> m_density;
		std::vector<unsigned int> m_particleId;
		std::vector<ParticleState> m_particleState;
		float m_V;
		float m_density0;
		unsigned int m_pointSetIndex;
		SurfaceTensionZorillaRitter2020* m_surfaceTension;
		ViscosityWeiler2018* m_viscosity;

		// time step
		int m_iterations = 0;
		int m_minIterations = 2;
		int m_maxIterations = 100;
		float m_maxError = 0.01;

		SimulationDataDFSPH m_simulationData;
		int m_counter = 0;
		int m_iterationsV = 0;
		bool m_enableDivergenceSolver = true;
		int m_maxIterationsV = 100;
		float m_maxErrorV = static_cast<float>(0.1);
	};
}

#endif // !DFSPH_SIMULATION_H