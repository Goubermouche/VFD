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

	class SurfaceTensionSolverDFSPH {
	public:
		SurfaceTensionSolverDFSPH(DFSPHSimulation* base);

		void OnUpdate();
		void Sort(const PointSet& pointSet);

	private:
		bool ClassifyParticleConfigurable(double com, int non, double d_offset = 0);
		std::vector<glm::vec3> GetSphereSamplesLookUp(int N, float supportRadius, int start, const std::vector<float>& vec3, int mod);

	private:
		DFSPHSimulation* m_Base;

		float m_SurfaceTension;

		int m_SamplesPerSecond;
		float m_SmoothingFactor;
		float m_Factor;
		float m_ParticleRadius;
		float m_NeighborParticleRadius;
		float m_ClassifierSlope;
		float m_ClassifierConstant;
		bool m_TemporalSmoothing;
		int m_SamplesPerStep;
		float m_ClassifierOffset;
		float m_PCANMix;
		float m_PCACMix;
		int m_CurvatureLimit;
		int m_SmoothPassCount;

		std::vector<float> m_SmoothedCurvature;
		std::vector<float> m_FinalCurvature;
		std::vector<float> m_DeltaFinalCurvature;
		std::vector<float> m_ClassifierInput;
		std::vector<glm::vec3> m_MonteCarloSurfaceNormals;
		std::vector<glm::vec3> m_MonteCarloSurfaceNormalsSmooth;
		std::vector<float> m_MonteCarloSurfaceCurvature;
		std::vector<float> m_MonteCarloSurfaceCurvatureSmooth;
		std::vector<float> m_ClassifierOutput;
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
		SurfaceTensionSolverDFSPH* m_SurfaceTensionSolver;
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