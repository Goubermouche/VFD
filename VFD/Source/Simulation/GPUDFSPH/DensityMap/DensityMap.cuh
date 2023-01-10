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
		using ContinuousFunction = std::function<float(const glm::vec3&)>;
		using SamplePredicate = std::function<bool(const glm::vec3&)>;

		DensityMap(const BoundingBox<glm::vec3>& domain, glm::uvec3 resolution);
		~DensityMap();

		void AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate = nullptr);
		float Interpolate(unsigned int fieldID, const glm::vec3& point, glm::vec3* gradient = nullptr) const;

		DensityMapDeviceData* GetDeviceData();
		const BoundingBox<glm::vec3>& GetBounds() const;
	private:
		glm::vec3 IndexToNodePosition(unsigned int i) const;
		unsigned int MultiToSingleIndex(const glm::uvec3& index) const;
		glm::uvec3 SingleToMultiIndex(const unsigned int index) const;

		BoundingBox<glm::vec3> CalculateSubDomain(const glm::uvec3& index) const;
		BoundingBox<glm::vec3> CalculateSubDomain(const unsigned int index) const;
		static std::array<float, 32> ShapeFunction(const glm::vec3& xi, std::array<std::array<float, 3>, 32>* gradient = nullptr);
	private:
		DensityMapDeviceData* d_DeviceData = nullptr;

		std::vector<std::vector<float>> m_Nodes;
		std::vector<std::vector<std::array<unsigned int, 32>>> m_Cells;
		std::vector<std::vector<unsigned int>> m_CellMap;

		thrust::device_vector<float> d_Nodes;
		thrust::device_vector<unsigned int> d_CellMap;
		thrust::device_vector<unsigned int> d_Cells;

		BoundingBox<glm::vec3> m_Domain;

		glm::uvec3 m_Resolution;
		glm::vec3 m_CellSize = { 0.0f, 0.0f, 0.0f };
		glm::vec3 m_CellSizeInverse = { 0.0f, 0.0f, 0.0f };

		unsigned int m_CellCount = 0u;
		unsigned int m_FieldCount = 0u;
	};
}

#endif // !DENSITY_MAP_CUH