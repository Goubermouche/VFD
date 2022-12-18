#ifndef DENSITY_MAP_2_CUH
#define DENSITY_MAP_2_CUH

#include <thrust/device_vector.h>

#include "DensityMapDeviceData.cuh"

namespace vfd
{
	struct DensityMap2
	{
		DensityMap2() = default;
		DensityMap2(const std::string& meshSourceFile);

		DensityMapDeviceData* GetDeviceData();
	private:
		thrust::device_vector<double> m_Nodes;
		thrust::device_vector<unsigned int> m_CellMap;
		thrust::device_vector<unsigned int> m_Cells;

		BoundingBox<glm::dvec3> m_Domain;

		glm::uvec3 m_Resolution;
		glm::dvec3 m_CellSize;
		glm::dvec3 m_CellSizeInverse;

		unsigned int m_FieldCount;
	};
}

#endif // !DENSITY_MAP_2_CUH