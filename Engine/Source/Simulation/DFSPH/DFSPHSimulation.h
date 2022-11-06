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

		float m_SurfaceTensionSolver;
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
		float m_MaxPressureSolverError;
		unsigned int m_PressureSolverIterations;
		std::vector<glm::vec3> m_vDiff;
		float m_tangentialDistanceFactor;
		float m_ViscositySolver;

		ConjugateFreeGradientSolver m_solver;
	};

	struct DFSPHSimulationDescription {

	};

	class StaticRigidBody;
	struct StaticRigidBodyDescription;
	typedef PrecomputedKernel<CubicKernel, 10000> PrecomputedCubicKernel;

	// TODO: make a GPU version
	class DFSPHSimulation : public RefCounted
	{
	public:
		DFSPHSimulation(const DFSPHSimulationDescription& desc);
		~DFSPHSimulation();

		void OnUpdate();
		void OnRenderTemp();
		void UpdateVMVelocity();

		inline const std::vector<StaticRigidBody*> GetRigidBodies() const {
			return m_RigidBodies;
		}

		inline const unsigned int GetParticleCount() {
			return m_ParticleCount;
		}

		inline const float GetParticleRadius() {
			return m_ParticleRadius;
		}

		inline const float GetDensity0() const {
			return m_Density0;
		}

		inline const float GetParticleSupportRadius() {
			return m_SupportRadius;
		}

		inline const float GetTimeStepSize() {
			return m_TimeStepSize;
		}

		inline const glm::vec3& GetParticlePosition(unsigned int i) const {
			return m_ParticlePositions[i];
		}

		inline glm::vec3& GetParticlePosition(unsigned int i) {
			return m_ParticlePositions[i];
		}

		inline const float GetParticleDensity(unsigned int i) const {
			return m_ParticleDensities[i];
		}

		inline float& GetParticleDensity(unsigned int i) {
			return m_ParticleDensities[i];
		}

		inline const glm::vec3& GetParticleVelocity(unsigned int i) const {
			return m_ParticleVelocities[i];
		}

		inline glm::vec3& GetParticleAcceleration(unsigned int i) {
			return m_ParticleAccelerations[i];
		}

		inline const float GetParticleMass(unsigned int i) const {
			return m_ParticleMasses[i];
		}

		inline const Scalar3f8& GetPrecalculatedVolumeGradientW(unsigned int i) const {
			return m_PrecalculatedVolumeGradientW[i];
		}

		inline const unsigned int GetPrecalculatedIndicesSamePhase(unsigned int i) const {
			return m_PrecalculatedIndicesSamePhase[i];
		}

		inline unsigned int NumberOfNeighbors(const unsigned int pointSetIndex, const unsigned int neighborPointSetIndex, const unsigned int index) const
		{
			return static_cast<unsigned int>(m_NeighborhoodSearch->GetPointSet(pointSetIndex).GetNeighborCount(neighborPointSetIndex, index));
		}

		inline unsigned int GetNeighbor(const unsigned int pointSetIndex, const unsigned int neighborPointSetIndex, const unsigned int index, const unsigned int k) const
		{
			return m_NeighborhoodSearch->GetPointSet(pointSetIndex).GetNeighbor(neighborPointSetIndex, index, k);
		}

		inline const unsigned int* GetNeighborList(const unsigned int pointSetIndex, const unsigned int neighborPointSetIndex, const unsigned int index) const
		{
			return m_NeighborhoodSearch->GetPointSet(pointSetIndex).GetNeighborList(neighborPointSetIndex, index).data();
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
		bool paused = true;

		// TODO: store everything things such as gravity in the description
	private:
		Ref<Material> m_Material;
		NeighborhoodSearch* m_NeighborhoodSearch;
		SurfaceTensionZorillaRitter2020* m_SurfaceTensionSolver;
		ViscosityWeiler2018* m_ViscositySolver;
		std::vector<StaticRigidBody*> m_RigidBodies;

		float m_ParticleRadius;
		float m_SupportRadius;
		float m_TimeStepSize;
		unsigned int m_ParticleCount;

		glm::vec3 m_Gravity;

		float m_CFLFactor;
		float m_CFLMinTimeStepSize;
		float m_CFLMaxTimeStepSize;

		// TODO: create a separate particle struct
		std::vector<float> m_Factor;
		std::vector<float> m_Kappa;
		std::vector<float> m_KappaVolume;
		std::vector<float> m_DensityAdvection;
		std::vector<Scalar3f8> m_PrecalculatedVolumeGradientW;
		std::vector<unsigned int> m_PrecalculatedIndices;
		std::vector<unsigned int> m_PrecalculatedIndicesSamePhase;
		std::vector<glm::vec3> m_ParticlePositions;
		std::vector<glm::vec3> m_ParticlePositions0;
		std::vector<glm::vec3> m_ParticleVelocities;
		std::vector<glm::vec3> m_ParticleVelocities0;
		std::vector<glm::vec3> m_ParticleAccelerations;
		std::vector<float> m_ParticleMasses;
		std::vector<float> m_ParticleDensities;

		float m_Volume;
		float m_Density0;
		float m_WZero;

		unsigned int m_PressureSolverIterations = 0;
		unsigned int m_MinPressureSolverIteratations = 2;
		unsigned int m_MaxPressureSolverIterations = 100;
		float m_MaxPressureSolverError = 0.01;

		unsigned int m_FrameCounter = 0;
		unsigned int m_VolumeSolverIterations = 0;

		bool m_EnableDivergenceSolver = true;
		unsigned int m_MaxVolumeSolverIterations = 100;
		float m_MaxVolumeError = static_cast<float>(0.1);
	};
}

#endif // !DFSPH_SIMULATION_H