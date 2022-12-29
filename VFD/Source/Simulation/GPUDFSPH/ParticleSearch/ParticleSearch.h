#ifndef PARTICLE_SEARCH_H
#define PARTICLE_SEARCH_H

#include "NeighborSet.h"
#include "SearchInfo.h"
#include "Simulation/GPUDFSPH/DFSPHParticle.h"
#include "Compute/ComputeHelper.h"
#include "Core/Structures/BoundingBox.h"

#include <thrust/device_vector.h>


namespace vfd
{
	class ParticleSearch
	{
	public:
		ParticleSearch(unsigned int pointCount, float searchRadius)
			: m_ParticleCount(pointCount), m_SearchRadius(searchRadius) {
			COMPUTE_SAFE(cudaMalloc(&d_NeighborSet, sizeof(NeighborSet)))

			unsigned int threadStarts = 0;
			m_ThreadsPerBlock = 64;
			ComputeHelper::GetThreadBlocks(m_ParticleCount, m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);
		}

		~ParticleSearch() {
			COMPUTE_SAFE(cudaFree(d_NeighborSet))
		}

		const NeighborSet* GetNeighborSet() const {
			return d_NeighborSet;
		}

		const thrust::device_vector<unsigned int>& GetSortIndices() const {
			return d_SortIndices;
		}

		void FindNeighbors(const DFSPHParticle* pointSet) {
			m_Particles = pointSet;

			if (m_ParticleCount == 0) {
				return;
			}

			// Update the point set
			ComputeMinMax();
			ComputeCellInformation();

			// Locate neighboring sets
			ComputeNeighborhood();
		}

		const BoundingBox<glm::vec3>& GetBounds() const
		{
			return m_Bounds;
		}

		void Sort(DFSPHParticle* particles);
	private:
		void ComputeMinMax();
		void ComputeCellInformation();
		void ComputeNeighborhood();
	private:
		NeighborSet* d_NeighborSet;
		SearchInfo m_GridInfo;
		BoundingBox<glm::vec3> m_Bounds;
		const DFSPHParticle* m_Particles;

		thrust::device_vector<glm::ivec3> d_MinMax;
		thrust::device_vector<unsigned int> d_TempSortIndices;
		thrust::device_vector<unsigned int> d_Neighbors;
		thrust::device_vector<unsigned int> d_NeighborCounts;
		thrust::device_vector<unsigned int> d_NeighborWriteOffsets;
		thrust::device_vector<unsigned int> d_ParticleCellIndices;
		thrust::device_vector<unsigned int> d_CellOffsets;
		thrust::device_vector<unsigned int> d_CellParticleCounts;
		thrust::device_vector<unsigned int> d_SortIndices;
		thrust::device_vector<unsigned int> d_ReversedSortIndices;

		unsigned int m_ThreadsPerBlock;
		unsigned int m_BlockStartsForParticles;
		unsigned int m_ParticleCount;

		float m_SearchRadius;
	};
}

#endif // !PARTICLE_SEARCH_H