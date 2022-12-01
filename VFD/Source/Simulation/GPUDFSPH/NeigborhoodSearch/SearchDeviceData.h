#ifndef CUN_SEARCH_DEVICE_DATA_H
#define CUN_SEARCH_DEVICE_DATA_H

#include "PointSet.h"
#include "Compute/Utility/Array.cuh"
#include "PointSetImplementation.h"
#include <thrust/device_vector.h>

namespace vfdcu {
	class SearchDeviceData {
	public:
		SearchDeviceData(float searchRadius)
		{
			m_SearchRadius = searchRadius;
		}

		void setSearchRadius(float searchRadius)
		{
			m_SearchRadius = searchRadius;
		}

		void ComputeMinMax(PointSet& pointSet);
		void ComputeCellInformation(PointSet& pointSet);
		void ComputeNeighborhood(PointSet& queryPointSet, PointSet& pointSet, unsigned int neighborListEntry);
	private:
		float m_SearchRadius;

		thrust::device_vector<glm::ivec3> d_MinMax;
		thrust::device_vector<unsigned int> d_TempSortIndices;
		thrust::device_vector<unsigned int> d_Neighbors;
		thrust::device_vector<unsigned int> d_NeighborCounts;
		thrust::device_vector<unsigned int> d_NeighborWriteOffsets;
	};
}

#endif // !CUN_SEARCH_DEVICE_DATA_H