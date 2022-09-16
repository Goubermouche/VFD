#ifndef SDF_H
#define SDF_H

#include "Core/Structures/BoundingBox.h"
#include "Renderer/Mesh/EdgeMesh.h"

namespace fe {
	class SDF : public RefCounted
	{
	public:
		using ContinuousFunction = std::function<float(const glm::vec3&)>;
		using SamplePredicate = std::function<bool(const glm::vec3&)>;

		SDF(const BoundingBox& domain, glm::ivec3 resolution);
		SDF(const EdgeMesh& mesh, const BoundingBox& bounds, const glm::uvec3& resolution, bool inverted = false);
		~SDF() = default;

		float GetDistance(const glm::vec3& point, float thickness) const;
		uint32_t AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate = nullptr);
		float Interpolate(unsigned int fieldID, const glm::vec3& point, glm::vec3* gradient = nullptr) const;
		const BoundingBox& GetDomain() const { return m_Domain; }
	private:
		glm::vec3 IndexToNodePosition(uint32_t index) const;

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

		std::vector<std::vector<float>> m_Nodes;
		std::vector<std::vector<std::array<uint32_t, 32>>> m_Cells;
		std::vector<std::vector<uint32_t>> m_CellMap;
	};
}

#endif // !SDF_H