#ifndef SDF_H
#define SDF_H

#include "Core/Structures/BoundingBox.h"
#include "Renderer/Mesh/EdgeMesh.h"

namespace fe {
	class SDF : public RefCounted
	{
	public:
		using ContinuousFunction = std::function<float(const glm::vec3&)>;

		SDF(const EdgeMesh& mesh, const BoundingBox& bounds, const glm::uvec3& resolution, bool inverted = false);
		~SDF() = default;

		float GetDistance(const glm::vec3& point, float thickness) const;
	private:
		uint32_t AddFunction(const ContinuousFunction& function);
		glm::vec3 IndexToNodePosition(uint32_t index) const;
		float Interpolate(const glm::vec3& point, glm::vec3* gradient = nullptr) const;

		glm::ivec3 SingleToMultiIndex(uint32_t index) const;
		uint32_t MultiToSingleIndex(const glm::ivec3& index) const;

		BoundingBox CalculateSubDomain(const glm::vec3& index) const;
		BoundingBox CalculateSubDomain(uint32_t index) const;

		static std::array<float, 32> ShapeFunction(const glm::vec3& xi, std::array<std::array<float, 3>, 32>* gradient = nullptr);
	private:
		BoundingBox m_Domain;

		uint32_t m_CellCount;
		uint32_t m_FieldCount;

		glm::uvec3 m_Resolution;
		glm::vec3 m_CellSize;
		glm::vec3 m_CellSizeInverse;

		std::vector<float> m_Nodes;
		std::vector<std::array<uint32_t, 32>> m_Cells;
		std::vector<uint32_t> m_CellMap;
	};
}

#endif // !SDF_H