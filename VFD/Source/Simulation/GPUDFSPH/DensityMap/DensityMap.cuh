#ifndef DENSITY_MAP_CUH
#define DENSITY_MAP_CUH

#include <thrust/device_vector.h>

#include "Simulation/GPUDFSPH/DensityMap/DensityMapDeviceData.cuh"
#include "Renderer/Mesh/EdgeMesh.h"
#include "Utility/SDF/MeshDistance.h"

namespace vfd
{
	struct DensityMap : public RefCounted
	{
		using ContinuousFunction = std::function<float(const glm::dvec3&)>;
		using SamplePredicate = std::function<bool(const glm::dvec3&)>;

		DensityMap(const BoundingBox<glm::dvec3>& domain, glm::uvec3 resolution);
		~DensityMap();

		void AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate = nullptr);
		double Interpolate(unsigned int fieldID, const glm::dvec3& point, glm::dvec3* gradient = nullptr);

		DensityMapDeviceData* GetDeviceData();
		const BoundingBox<glm::dvec3>& GetBounds() const;
	private:
		glm::dvec3 IndexToNodePosition(unsigned int i) const;
		unsigned int MultiToSingleIndex(const glm::uvec3& index) const;
		glm::uvec3 SingleToMultiIndex(const unsigned int index) const;

		BoundingBox<glm::dvec3> CalculateSubDomain(const glm::uvec3& index) const;
		BoundingBox<glm::dvec3> CalculateSubDomain(const unsigned int index) const;
		static std::array<double, 32> ShapeFunction(const glm::dvec3& xi, std::array<std::array<double, 3>, 32>* gradient = nullptr);
	private:
		DensityMapDeviceData* d_DeviceData = nullptr;

		std::vector<std::vector<double>> m_Nodes;
		std::vector<std::vector<std::array<unsigned int, 32>>> m_Cells;
		std::vector<std::vector<unsigned int>> m_CellMap;

		thrust::device_vector<double> d_Nodes;
		thrust::device_vector<unsigned int> d_CellMap;
		thrust::device_vector<unsigned int> d_Cells;

		BoundingBox<glm::dvec3> m_Domain;

		glm::uvec3 m_Resolution;
		glm::dvec3 m_CellSize;
		glm::dvec3 m_CellSizeInverse;

		unsigned int m_CellCount = 0u;
		unsigned int m_FieldCount = 0u;
	};
}

#endif // !DENSITY_MAP_CUH