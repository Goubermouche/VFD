#ifndef DFSPH_IMPLEMENTATION_H
#define DFSPH_IMPLEMENTATION_H

#include "pch.h"

#include "Renderer/VertexArray.h"
#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "DFSPHKernels.cuh"
#include "GPUDFSPHSimulationDescription.h"
#include "RigidBody/RigidBody.cuh"
#include "ParticleSearch/ParticleSearch.h"

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
	__host__ __device__	float operator()(const float& x) const {
		return x * x;
	}
};

struct DotUnaryOperator 
{
	__host__ __device__	float operator()(thrust::tuple<float, float> tuple) const {
		return  thrust::get<0>(tuple) * thrust::get<1>(tuple);
	}
};

namespace vfd
{
	class DFSPHImplementation : public RefCounted
	{
	public:
		DFSPHImplementation(const GPUDFSPHSimulationDescription& desc, std::vector<Ref<RigidBody>>& rigidBodies);
		~DFSPHImplementation();

		void OnUpdate();
		void Reset(); // DEBUG

		// Getters 
		const Ref<VertexArray>& GetVertexArray() const;
		unsigned int GetParticleCount() const;
		float GetParticleRadius() const;
		float GetMaxVelocityMagnitude() const;
		float GetTimeStepSize() const;
		const ParticleSearch* GetParticleSearch() const;
		const GPUDFSPHSimulationDescription& GetDescription() const;
		void SetDescription(const GPUDFSPHSimulationDescription& desc);
	private:
		/// <summary>
		/// Initializes the rigid body objects currently present in the scene, 
		///	TODO: Add parameters.
		/// </summary>
		void InitRigidBodies(std::vector<Ref<RigidBody>>& rigidBodies);

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
	private:
		DFSPHParticle* m_Particles = nullptr;

		// TODO: use a thrust::device_vector for multiple rigid bodies
		RigidBodyDeviceData* d_RigidBodyData = nullptr;

		// Simulation info
		DFSPHSimulationInfo m_Info;
		DFSPHSimulationInfo* d_Info = nullptr;
		GPUDFSPHSimulationDescription m_Description;

		// Viscosity: TODO move to a separate generic class?
		//            TODO move to the particle struct?
		thrust::device_vector<glm::mat3x3> m_PreconditionerInverseDiagonal;
		thrust::device_vector<float> m_ViscosityGradientB;
		thrust::device_vector<float> m_ViscosityGradientG;
		thrust::device_vector<float> m_Preconditioner;
		thrust::device_vector<float> m_PreconditionerZ;
		thrust::device_vector<float> m_Residual;
		thrust::device_vector<float> m_OperationTemporary;
		thrust::device_vector<float> m_Temp;

		// Neighborhood search
		const NeighborSet* d_NeighborSet = nullptr;
		ParticleSearch* m_ParticleSearch = nullptr;

		// Unary operators 
		MaxVelocityMagnitudeUnaryOperator m_MaxVelocityMagnitudeUnaryOperator;
		DensityErrorUnaryOperator m_DensityErrorUnaryOperator;
		SquaredNormUnaryOperator m_SquaredNormUnaryOperator;
		DotUnaryOperator m_DotUnaryOperator;

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