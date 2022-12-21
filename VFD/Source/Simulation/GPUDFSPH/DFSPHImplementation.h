#ifndef DFSPH_IMPLEMENTATION_H
#define DFSPH_IMPLEMENTATION_H

#include "pch.h"

#include "Renderer/VertexArray.h"
#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "NeigborhoodSearch/NeighborhoodSearchP.h"
#include "DFSPHKernels.cuh"
#include "GPUDFSPHSimulationDescription.h"
#include "RigidBody/RigidBody.cuh"
#include "Scalar/Vec3Vec8.h"

#include <thrust\device_vector.h>

struct MaxVelocityMagnitudeUnaryOperator
{
	float TimeStepSize;

	// Calculates the velocity magnitude of a given particle using the provided time step size
	__host__ __device__	float operator()(const vfd::DFSPHParticle& x) const {
		return glm::length2(x.Velocity + x.Acceleration * TimeStepSize);
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
		float GetMaxVelocityMagnitude() const;
		float GetTimeStepSize() const;

		unsigned int GetNumberOfNeighbors(const unsigned int index) const;
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
		void CalculateTimeStepSize(const thrust::device_ptr<DFSPHParticle>& mappedParticles);

		/// <summary>
		/// Calculates the highest velocity magnitude in the simulation
		/// </summary>
		void CalculateMaxVelocityMagnitude(const thrust::device_ptr<DFSPHParticle>& mappedParticles, float initialValue);

		//void PrecomputeValues(DFSPHParticle* particles);
	private:
		DFSPHParticle* m_Particles = nullptr;
		DFSPHParticle0* m_Particles0 = nullptr;

		RigidBodyDeviceData* d_RigidBodyData = nullptr;
		PointSetDeviceData* d_PointSet = nullptr;

		DFSPHSimulationInfo m_Info;
		DFSPHSimulationInfo* d_Info = nullptr;
		GPUDFSPHSimulationDescription m_Description;

		NeighborhoodSearch* m_NeighborhoodSearch = nullptr;
		MaxVelocityMagnitudeUnaryOperator m_MaxVelocityMagnitudeUnaryOperator;

		//std::vector<unsigned int> m_PreCalculatedIndices;
		//std::vector<unsigned int> m_PreCalculatedIndicesSamePhase;
		//std::vector<vec3vec8> m_PrecalculatedVolumeGradientW;

		//thrust::device_vector<unsigned int> d_PreCalculatedIndices;
		//thrust::device_vector<unsigned int> d_PreCalculatedIndicesSamePhase;
		//thrust::device_vector<vec3vec8> d_PrecalculatedVolumeGradientW;

		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;
			
		unsigned int m_IterationCount = 0;
		float m_MaxVelocityMagnitude;

		int m_ThreadsPerBlock = MAX_CUDA_THREADS_PER_BLOCK;
		unsigned int m_BlockStartsForParticles;
	};
}

#endif // !DFSPH_IMPLEMENTATION_H