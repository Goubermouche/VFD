#ifndef POINT_SET_IMPLEMENTATION_P_H
#define POINT_SET_IMPLEMENTATION_P_H

#include "GridInfoP.h"
#include <thrust/device_vector.h>
#include "Compute/ComputeHelper.h"
#include "../DFSPHParticle.h"

namespace vfd {
	class NeighborhoodSearch;
	class SearchDeviceData;

	class PointSetImplementation {
	public:
		PointSetImplementation(size_t particleCount, DFSPHParticle* particles);

		PointSetImplementation(PointSetImplementation const& other) = default;
		PointSetImplementation& operator=(PointSetImplementation const& other);
		~PointSetImplementation() { }

		void Resize(size_t particleCount, DFSPHParticle* particles)
		{
			m_ParticleCount = particleCount;
			m_Particles = particles;

			unsigned int threadStarts = 0;
			vfd::ComputeHelper::GetThreadBlocks(static_cast<unsigned int>(particleCount), m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);

			CopyToDevice();
		}

		void SortField(DFSPHParticle* particles);

		void CopyToDevice();
	private:
		friend NeighborhoodSearch;
		friend SearchDeviceData;

		// Min Max of all particles
		glm::vec3 m_Min;
		glm::vec3 m_Max;

		size_t m_ParticleCount;
		DFSPHParticle* m_Particles;
		int m_ThreadsPerBlock;
		unsigned int m_BlockStartsForParticles;
		GridInfo m_GridInfo;

		thrust::device_vector<DFSPHParticle> d_Particles;
		thrust::device_vector<unsigned int> d_ParticleCellIndices;
		thrust::device_vector<unsigned int> d_CellOffsets;
		thrust::device_vector<unsigned int> d_CellParticleCounts;
		thrust::device_vector<unsigned int> d_SortIndices;
		thrust::device_vector<unsigned int> d_ReversedSortIndices;

		void PrepareInternalDataStructures(GridInfo& gridInfo, size_t numberOfCells);
	};
}

#endif // !POINT_SET_IMPLEMENTATION_H