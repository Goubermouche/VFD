#ifndef DFSPH_SIMULATION_H
#define DFSPH_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Simulation/DFSPH/CompactNSearch.h"
#include "Simulation/DFSPH/StaticBoundarySimulator.h"
#include "Kernel.h"
#include "MatrixFreeSolver.h"
#include "Core/Math/Scalar3f8.h"

// Inspired by: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH

namespace fe {
	class DFSPHSimulation;

	class SurfaceTensionZorillaRitter2020 {
	public:
		SurfaceTensionZorillaRitter2020(DFSPHSimulation* base);
		void OnUpdate();
		bool ClassifyParticleConfigurable(double com, int non, double d_offset = 0);
		std::vector<glm::vec3> GetSphereSamplesLookUp(int N, float supportRadius, int start, const std::vector<float>& vec3, int mod);

		DFSPHSimulation* m_Base;
		static int PCA_NRM_MODE;
		static int PCA_NRM_MIX;
		static int PCA_CUR_MIX;
		static int FIX_SAMPLES;
		static int NEIGH_LIMIT;

		static int SAMPLING;
		static int SAMPLING_HALTON;
		static int SAMPLING_RND;

		static int NORMAL_MODE;
		static int NORMAL_PCA;
		static int NORMAL_MC;
		static int NORMAL_MIX;

		static int SMOOTH_PASSES;
		static int TEMPORAL_SMOOTH;

		int  m_Csd;       // number of samples per particle per second
		float m_tau;       // smoothing factor, default 0.5
		float m_r2mult;    // r1 to R2 factor, default 0.8
		float m_r1;        // radius of current particle
		float m_r2;        // radius of neighbor particles
		float m_class_k;   // slope of the linear classifier
		float m_class_d;   // constant of the linear classifier
		bool m_temporal_smoothing;
		int    m_CsdFix;            // number of samples per computational step
		float   m_class_d_off;       // offset of classifier d used for PCA neighbors
		float   m_pca_N_mix;         // mixing factor of PCA normal and MC normal
		float   m_pca_C_mix;         // mixing factor of PCA curvature and MC curvature
		int    m_neighs_limit;      // maximum nr of neighbors used in PCA computation
		int    m_CS_smooth_passes;  // nr of smoohting passes

		std::vector<glm::vec3> m_pca_normals;       // surface normal by PCA
		std::vector<float>     m_pca_curv;          // curvature estimate by spherity
		std::vector<float>     m_pca_curv_smooth;   // smoothed curvature
		std::vector<float>     m_final_curvatures;
		std::vector<float>     m_final_curvatures_old;
		std::vector<float>     m_classifier_input;

		std::vector<glm::vec3> m_mc_normals;          // Monte Carlo surface normals
		std::vector<glm::vec3> m_mc_normals_smooth;   // smoothed normals
		std::vector<float>     m_mc_curv;             // Monte Carlo surface curvature
		std::vector<float>     m_mc_curv_smooth;      // smoothed curvature
		std::vector<float>     m_classifier_output;   // outut of the surface classifier

		float m_surfaceTension;
		float m_surfaceTensionBoundary;
	};

	class ViscosityWeiler2018 {
	public:
		static void MatrixVecProd(const std::vector<float>&, std::vector<float>& result, void* userData, DFSPHSimulation* sim);

		ViscosityWeiler2018(DFSPHSimulation* base);
		void OnUpdate();
		static void DiagonalMatrixElement(const unsigned int i, glm::mat3x3& result, void* userData, DFSPHSimulation* m_Base);
		void ComputeRHS(std::vector<float>& b, std::vector<float>& g);
		void ApplyForces(const std::vector<float>& x);

		DFSPHSimulation* m_Base;
		float m_boundaryViscosity;
		unsigned int m_maxIter;
		float m_maxError;
		unsigned int m_iterations;
		std::vector<glm::vec3> m_vDiff;
		float m_tangentialDistanceFactor;
		float m_viscosity;

		ConjugateFreeGradientSolver m_solver;
	};

	class FluidModel {
	public:

	};

	struct DFSPHSimulationDescription {

	};

	struct FluidData {
		std::string id;
		std::string samplesFile;
		std::string visMeshFile;
		glm::vec3 Position;
		glm::mat4 Rotation;
		glm::vec3 Scale;
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

	class SimulationDataDFSPH {
	public:
		SimulationDataDFSPH();
		void Init(DFSPHSimulation* sim);

		void PerformNeighborhoodSearchSort(DFSPHSimulation* base);

		inline const float GetFactor(const unsigned int fluidIndex, const unsigned int i) const {
			return m_factor[fluidIndex][i];
		}

		inline float& GetFactor(const unsigned int fluidIndex, const unsigned int i) {
			return m_factor[fluidIndex][i];
		}

		inline float& GetDensityAdv(const unsigned int fluidIndex, const unsigned int i) {
			return m_density_adv[fluidIndex][i];
		}

		inline float& GetKappaV(const unsigned int fluidIndex, const unsigned int i)
		{
			return m_kappaV[fluidIndex][i];
		}

		inline void SetKappaV(const unsigned int fluidIndex, const unsigned int i, const float p)
		{
			m_kappaV[fluidIndex][i] = p;
		}

		inline const float GetKappa(const unsigned int fluidIndex, const unsigned int i) const
		{
			return m_kappa[fluidIndex][i];
		}

		inline float& GetKappa(const unsigned int fluidIndex, const unsigned int i)
		{
			return m_kappa[fluidIndex][i];
		}
	protected:
		std::vector<std::vector<float>> m_factor;
		std::vector<std::vector<float>> m_kappa;
		std::vector<std::vector<float>> m_kappaV;
		std::vector<std::vector<float>> m_density_adv;
	};

	class StaticRigidBody;
	struct StaticRigidBodyDescription;

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
		void UpdateVMVelocity();

		inline unsigned int NumberOfNeighbors(const unsigned int pointSetIndex, const unsigned int neighborPointSetIndex, const unsigned int index) const
		{
			return static_cast<unsigned int>(m_neighborhoodSearch->GetPointSet(pointSetIndex).GetNeighborCount(neighborPointSetIndex, index));
		}

		inline unsigned int GetNeighbor(const unsigned int pointSetIndex, const unsigned int neighborPointSetIndex, const unsigned int index, const unsigned int k) const
		{
			return m_neighborhoodSearch->GetPointSet(pointSetIndex).GetNeighbor(neighborPointSetIndex, index, k);
		}

		inline const unsigned int* GetNeighborList(const unsigned int pointSetIndex, const unsigned int neighborPointSetIndex, const unsigned int index) const
		{
			return m_neighborhoodSearch->GetPointSet(pointSetIndex).GetNeighborList(neighborPointSetIndex, index).data();
		}

		FluidModel* GetFluidModelFromPointSet(const unsigned int pointSetIndex) {
			return static_cast<FluidModel*>(m_neighborhoodSearch->GetPointSet(pointSetIndex).GetUserData());
		}
	private:
		void SetParticleRadius(float val);
		void ComputeVolumeAndBoundaryX();
		void ComputeVolumeAndBoundaryX(const unsigned int i, const glm::vec3& xi);
		void ComputeDensities();
		void ComputeDFSPHFactor();
		void DivergenceSolve();
		void WarmStartDivergenceSolve();
		void ComputeDensityChange(const unsigned int i, const float h);
		void DivergenceSolveIteration(float& avg_density_err);
		void ClearAccelerations();
		void ComputeNonPressureForces();
		void UpdateTimeStepSize();
		void PressureSolve();
		void WarmStartPressureSolve();
		void ComputeDensityAdv(const unsigned int i, const int numParticles, const float h, const float density0);
		void PressureSolveIteration(float& avg_density_err);
		void PrecomputeValues();

		void InitFluidData();
	public:
		Ref<Material> m_Material;
		bool paused = true;
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
		std::vector<FluidData*> fluidModels;
		std::vector<MaterialData*> materials;
		float particleRadius;
		float timeStepSize;

		// sim 
		float m_supportRadius;
		
		bool m_enableZSort;
		glm::vec3 m_gravitation;
		float m_cflFactor;
		float m_cflMinTimeStepSize;
		float m_cflMaxTimeStepSize;
		NeighborhoodSearch* m_neighborhoodSearch;
		bool m_simulationIsInitialized;
		const float m_eps = static_cast<float>(1.0e-5);

		// fluid model
		enum class ParticleState { Active = 0, AnimatedByEmitter, Fixed };
		int m_numParticles;
		unsigned int m_numActiveParticles;
		unsigned int m_numActiveParticles0;

		std::vector<Scalar3f8> m_precomp_V_gradW;
		std::vector<unsigned int> m_precompIndices;
		std::vector<unsigned int> m_precompIndicesSamePhase;

		std::vector<glm::vec3> m_Position;
		std::vector<glm::vec3> m_x0;
		std::vector<glm::vec3> m_v;
		std::vector<glm::vec3> m_v0;
		std::vector<glm::vec3> m_a;
		std::vector<float> m_masses;
		std::vector<float> m_density;
		std::vector<unsigned int> m_particleId;
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
		float m_W_zero;

		std::vector<StaticRigidBody*> m_RigidBodies;
	};
}

#endif // !DFSPH_SIMULATION_H