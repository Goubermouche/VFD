#ifndef POINT_SET_IMPLEMENTATION_H
#define POINT_SET_IMPLEMENTATION_H

#include "GridInfo.h"
#include "Compute/Utility/Array.cuh"
#include <thrust/device_vector.h>
#include "Utils/cuda_helper.h"

namespace vfdcu {
	class NeighborhoodSearch;
	class SearchDeviceData;

	class PointSetImplementation {
	public:
		PointSetImplementation(size_t particleCount, glm::vec3* particles);

		PointSetImplementation(PointSetImplementation const& other) = default;
		PointSetImplementation& operator=(PointSetImplementation const& other);
		~PointSetImplementation() { }

		void Resize(size_t particleCount, glm::vec3* particles)
		{
			m_ParticleCount = particleCount;
			m_Particles = particles;

			unsigned int threadStarts = 0;
			CudaHelper::GetThreadBlocks(static_cast<unsigned int>(particleCount), m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);

			CopyToDevice();
		}

		void CopyToDevice();
	private:
		friend NeighborhoodSearch;
		friend SearchDeviceData;

		// Min Max of all particles
		glm::vec3 m_Min;
		glm::vec3 m_Max;

		size_t m_ParticleCount;
		glm::vec3* m_Particles;
		int m_ThreadsPerBlock;
		unsigned int m_BlockStartsForParticles;
		GridInfo m_GridInfo;

		thrust::device_vector<glm::vec3> d_Particles;
		thrust::device_vector<unsigned int> d_ParticleCellIndices;
		thrust::device_vector<unsigned int> d_CellOffsets;
		thrust::device_vector<unsigned int> d_CellParticleCounts;
		thrust::device_vector<unsigned int> d_SortIndices;
		thrust::device_vector<unsigned int> d_ReversedSortIndices;

		void PrepareInternalDataStructures(GridInfo& gridInfo, size_t numberOfCells);
	};
}

#endif // !POINT_SET_IMPLEMENTATION_H