#ifndef DFSPH_IMPLEMENTATION_H
#define DFSPH_IMPLEMENTATION_H

#include "pch.h"

#include "Renderer/VertexArray.h"
#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "DFSPHKernels.cuh"
#include "DFSPHSimulationDescription.h"
#include "RigidBody/RigidBody.cuh"
#include "ParticleSearch/ParticleSearch.h"
#include "Debug/Timer.h"

#include <thrust/device_vector.h>

struct MaxVelocityMagnitudeUnaryOperator
{
	float TimeStepSize;

	// Calculates the velocity magnitude of a given particle using the provided time step size
	__host__ __device__	float operator()(const vfd::DFSPHParticle& x) const {
		return glm::length2(x.Velocity + x.Acceleration * TimeStepSize);
	}
};

struct DensityErrorUnaryOperator
{
	float Density0;

	__host__ __device__	float operator()(const vfd::DFSPHParticle& x) const	{
		return Density0 * x.PressureResiduum;
	}
};

struct SquaredNormUnaryOperator
{
	__host__ __device__	float operator()(const glm::vec3& x) const {
		return glm::compAdd(x * x);
	}
};

struct DotUnaryOperator 
{
	__host__ __device__	float operator()(thrust::tuple<glm::vec3, glm::vec3> tuple) const {
		return  glm::compAdd(thrust::get<0>(tuple) * thrust::get<1>(tuple));
	}
};

struct Vec3FloatMultiplyBinaryOperator
{
	__host__ __device__	glm::vec3 operator()(const glm::vec3& x, const float& y) const
	{
		return x * y;
	}
};

struct Vec3Mat3MultiplyBinaryOperator
{
	__host__ __device__	glm::vec3 operator()(const glm::mat3x3& x, const glm::vec3& y) const
	{
		return x * y;
	}
};


namespace vfd
{
	struct DFSPHDebugInfo
	{
		Timer NeighborhoodSearchTimer;
		Timer BaseSolverTimer;
		Timer DivergenceSolverTimer;
		Timer SurfaceTensionSolverTimer;
		Timer ViscositySolverTimer;
		Timer PressureSolverTimer;
	};

	class DFSPHImplementation : public RefCounted
	{
	public:
		DFSPHImplementation(const DFSPHSimulationDescription& desc);
		~DFSPHImplementation();

		void Simulate(const std::vector<Ref<RigidBody>>& rigidBodies);
		void OnUpdate();
		void Reset(); // DEBUG

		// Getters 
		const Ref<VertexArray>& GetVertexArray() const;
		unsigned int GetParticleCount() const;
		float GetParticleRadius() const;
		float GetMaxVelocityMagnitude() const;
		float GetTimeStepSize() const;
		const ParticleSearch* GetParticleSearch() const;
		const DFSPHSimulationDescription& GetDescription() const;
		void SetDescription(const DFSPHSimulationDescription& desc);
		const DFSPHSimulationInfo& GetInfo() const;
		PrecomputedDFSPHCubicKernel& GetKernel();
		const std::vector<Ref<RigidBody>>& GetRigidBodies() const;
		const DFSPHDebugInfo& GetDebugInfo() const;
	private:
		/// <summary>
		/// Initializes the rigid body objects currently present in the scene, 
		///	TODO: Add parameters.
		/// </summary>
		// void InitRigidBodies();

		/// <summary>
		/// Initializes the particles and their data.
		///	TODO: Add parameters.
		/// </summary>
		void InitFluidData();

		/// <summary>
		/// Updates the simulation time step size based on the highest velocity magnitude.
		/// </summary>
		/// <param name="mappedParticles">Specifies the particle array that contains the max velocity magnitude</param>
		void ComputeTimeStepSize(const thrust::device_ptr<DFSPHParticle>& mappedParticles);

		/// <summary>
		/// Calculates the highest velocity magnitude in the simulation
		/// </summary>
		void ComputeMaxVelocityMagnitude(const thrust::device_ptr<DFSPHParticle>& mappedParticles, float initialValue);

		void ComputePressure(DFSPHParticle* particles);
		void ComputeDivergence(DFSPHParticle* particles);
		void ComputeViscosity(DFSPHParticle* particles);

		void SolveViscosity(DFSPHParticle* particles);
		void SolveSurfaceTension(DFSPHParticle* particles);
	private:
		DFSPHParticle* m_Particles = nullptr;

		// Rigid bodies
		std::vector<Ref<RigidBody>> m_RigidBodies;
		thrust::device_vector<RigidBodyDeviceData*> d_RigidBodies;

		// Simulation info
		DFSPHSimulationInfo m_Info;
		DFSPHSimulationInfo* d_Info = nullptr;
		DFSPHSimulationDescription m_Description;
		DFSPHDebugInfo m_TimingData;

		// Viscosity: TODO move to a separate generic class?
		//            TODO move to the particle struct?
		thrust::device_vector<glm::mat3x3> m_PreconditionerInverseDiagonal;
		thrust::device_vector<glm::vec3> m_ViscosityGradientB;
		thrust::device_vector<glm::vec3> m_ViscosityGradientG;
		thrust::device_vector<glm::vec3> m_Preconditioner;
		thrust::device_vector<glm::vec3> m_PreconditionerZ;
		thrust::device_vector<glm::vec3> m_Residual;
		thrust::device_vector<glm::vec3> m_OperationTemporary;
		thrust::device_vector<glm::vec3> m_Temp;

		// Neighborhood search
		const NeighborSet* d_NeighborSet = nullptr;
		ParticleSearch* m_ParticleSearch = nullptr;

		// Unary operators 
		MaxVelocityMagnitudeUnaryOperator m_MaxVelocityMagnitudeUnaryOperator;
		DensityErrorUnaryOperator m_DensityErrorUnaryOperator;
		SquaredNormUnaryOperator m_SquaredNormUnaryOperator;
		DotUnaryOperator m_DotUnaryOperator;

		Vec3FloatMultiplyBinaryOperator m_Vec3FloatMultiplyBinaryOperator;
		Vec3Mat3MultiplyBinaryOperator m_Vec3Mat3MultiplyBinaryOperator;

		// Smoothing kernels
		PrecomputedDFSPHCubicKernel m_PrecomputedSmoothingKernel;
		PrecomputedDFSPHCubicKernel* d_PrecomputedSmoothingKernel;

		// OpenGL data
		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;
			
		unsigned int m_IterationCount = 0u;
		unsigned int m_DivergenceSolverIterationCount = 0u;
		unsigned int m_PressureSolverIterationCount = 0u;
		unsigned int m_ViscositySolverIterationCount = 0u;

		float m_DivergenceSolverError = 0.0f;
		float m_PressureSolverError = 0.0f;
		float m_ViscositySolverError = 0.0f;
		float m_MaxVelocityMagnitude = 0.0f;

		int m_ThreadsPerBlock = MAX_CUDA_THREADS_PER_BLOCK;
		unsigned int m_BlockStartsForParticles;
	};
}

#endif // !DFSPH_IMPLEMENTATION_H