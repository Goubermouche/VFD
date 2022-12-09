#ifndef DFSPH_IMPLEMENTATION_H
#define DFSPH_IMPLEMENTATION_H

#include "pch.h"
#include "Renderer/VertexArray.h"
#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "NeigborhoodSearch/NeighborhoodSearchP.h"
#include "DFSPHKernels.cuh"

namespace vfd
{
	class DFSPHImplementation : public RefCounted
	{
	public:
		DFSPHImplementation();
		~DFSPHImplementation();

		const Ref<VertexArray>& GetVertexArray() const;

		void OnUpdate();

		unsigned int GetParticleCount()
		{
			return m_Info.ParticleCount;
		}

		float GetParticleRadius()
		{
			return m_Info.ParticleRadius;
		}
	private:
		void InitFluidData();

		/// <summary>
		/// Updates the simulation time step size based on the highest velocity magnitude.
		/// </summary>
		/// <param name="mappedParticles">Specifies the particle array that contains the max velocity magnitude</param>
		void CalculateTimeStepSize(DFSPHParticle* mappedParticles);
	private:
		DFSPHParticle* m_Particles = nullptr;

		// RigidBody
		// {
		//     Position  
		//     Rotation  
		//     Scale     
		//     CollisionMap
		//     {
		//         glm::uvec3 Resolution
		//         glm::dvec3 CellSize
		//         glm::dvec3 CellSizeInverse
		//         size_t CellCount
		//         size_t FieldCount
		//         double* Nodes
		//         ?* cells
		//         unsigned int* cellMap
		//     }
		// }

		DFSPHSimulationInfo m_Info; 
		DFSPHSimulationInfo* d_Info = nullptr;


		NeighborhoodSearch* m_NeighborhoodSearch = nullptr;

		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;

		unsigned int m_IterationCount = 0;
		float m_CFLMaxTimeStepSize = 0.005f;
		float m_CFLMinTimeStepSize = 0.0001f;

		int m_ThreadsPerBlock = MAX_CUDA_THREADS_PER_BLOCK;
		unsigned int m_BlockStartsForParticles;
		float* m_TempReduction = nullptr;
		float* d_TempReduction = nullptr;
	};
}

#endif // !DFSPH_IMPLEMENTATION_H