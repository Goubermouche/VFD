#ifndef DFSPH_IMPLEMENTATION_H
#define DFSPH_IMPLEMENTATION_H

#include "pch.h"

#include "Renderer/VertexArray.h"

#include "Simulation/DFSPH/Structures/DFSPHParticle.h"
#include "Simulation/DFSPH/Structures/DFSPHSimulationInfo.h"
#include "Simulation/DFSPH/DFSPHKernels.cuh"
#include "Simulation/DFSPH/Structures/DFSPHSimulationDescription.h"
#include "Simulation/DFSPH/Structures/DFSPHFunctionObjects.h"
#include "Simulation/DFSPH/RigidBody/RigidBody.cuh"
#include "Simulation/DFSPH/ParticleSearch/ParticleSearch.h"
#include "Simulation/DFSPH/Structures/DFSPHDebugInfo.h"
#include "Simulation/DFSPH/ParticleBuffer/DFSPHParticleBuffer.h"

#include <thrust/device_vector.h>


namespace vfd
{
	class DFSPHImplementation : public RefCounted
	{
	public:

		enum class SimulationState
		{
			None,
			Simulating,
			Ready
		};

		DFSPHImplementation(const DFSPHSimulationDescription& desc);
		~DFSPHImplementation();

		void Simulate();
		void OnUpdate();

		// Setters
		void SetFluidObjects(const std::vector<Ref<FluidObject>>& fluidObjects);
		void SetRigidBodies(const std::vector<Ref<RigidBody>>& rigidBodies);

		// Getters
		SimulationState GetSimulationState() const;
		const Ref<VertexArray>& GetVertexArray() const;
		unsigned int GetParticleCount() const;
		float GetParticleRadius() const;
		float GetMaxVelocityMagnitude() const;
		float GetCurrentTimeStepSize() const;
		const ParticleSearch& GetParticleSearch() const;
		const DFSPHSimulationDescription& GetDescription() const;
		void SetDescription(const DFSPHSimulationDescription& desc);
		const DFSPHSimulationInfo& GetInfo() const;
		PrecomputedDFSPHCubicKernel& GetKernel();
		const std::vector<Ref<RigidBody>>& GetRigidBodies() const;
		unsigned int GetRigidBodyCount() const;
		const DFSPHDebugInfo& GetDebugInfo() const;
		Ref<DFSPHParticleBuffer> GetParticleFrameBuffer();
	private:
		/// <summary>
		/// Initializes the rigid body objects currently present in the scene, 
		///	TODO: Add parameters.
		/// </summary>
		// void InitRigidBodies();

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
		DFSPHParticle* d_Particles = nullptr;
		DFSPHParticleSimple* d_FrameBuffer = nullptr;
		Ref<DFSPHParticleBuffer> m_ParticleFrameBuffer;

		// Rigid bodies
		std::vector<Ref<RigidBody>> m_RigidBodies;
		thrust::device_vector<RigidBodyDeviceData*> d_RigidBodies;

		// Simulation info
		DFSPHSimulationInfo m_Info;
		DFSPHSimulationInfo* d_Info = nullptr;
		DFSPHSimulationDescription m_Description;
		DFSPHDebugInfo m_DebugInfo;

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
		ParticleSearch m_ParticleSearch;

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
		// Ref<VertexArray> m_VertexArray;
		// Ref<VertexBuffer> m_VertexBuffer;
			
		float m_MaxVelocityMagnitude = 0.0f;
		SimulationState m_State = SimulationState::None;

		int m_ThreadsPerBlock = MAX_CUDA_THREADS_PER_BLOCK;
		unsigned int m_BlockStartsForParticles;
	};
}

#endif // !DFSPH_IMPLEMENTATION_H