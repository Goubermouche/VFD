#ifndef SDF_H
#define SDF_H

#include "Core/Structures/BoundingBox.h"
#include "Renderer/Mesh/EdgeMesh.h"

namespace vfd {
	// TEMP
	template<class T>
	bool read(std::streambuf& buf, T& val)
	{
		static_assert(std::is_standard_layout<T>{}, "data is not standard layout");
		auto bytes = sizeof(T);
		return buf.sgetn(reinterpret_cast<char*>(&val), bytes) == bytes;
	}

	class SDF : public RefCounted
	{
	public:
		using ContinuousFunction = std::function<float(const glm::vec3&)>;
		using SamplePredicate = std::function<bool(const glm::vec3&)>;

		SDF(const BoundingBox<glm::vec3>& domain, glm::ivec3 resolution);
		SDF(const Ref<EdgeMesh>& mesh, const BoundingBox<glm::vec3>& bounds, const glm::uvec3& resolution, bool inverted = false);
		SDF(const std::string& filepath);
		~SDF() = default;

		float GetDistance(const glm::vec3& point, float thickness) const;
		unsigned int AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate = nullptr);

		float Interpolate(unsigned int fieldID, const glm::vec3& point, glm::vec3* gradient = nullptr) const;
		float Interpolate(unsigned int fieldID, const glm::vec3& xi, const std::array<unsigned int, 32>& cell, const glm::vec3& c0, const std::array<float, 32>& N,
			glm::vec3* gradient = nullptr, std::array<std::array<float, 3>, 32>* dN = nullptr);

		const BoundingBox<glm::vec3>& GetDomain() const { return m_Domain; }
		bool DetermineShapeFunctions(unsigned int fieldID, const glm::vec3& x, std::array<unsigned int, 32>& cell,
			glm::vec3& c0, std::array<float, 32>& N, std::array<std::array<float, 3>, 32>* dN = nullptr);
	private:
		glm::vec3 IndexToNodePosition(unsigned int index) const;

		glm::uvec3 SingleToMultiIndex(unsigned int index) const;
		unsigned int MultiToSingleIndex(const glm::uvec3& index) const;

		BoundingBox<glm::vec3> CalculateSubDomain(const glm::uvec3& index) const;
		BoundingBox<glm::vec3> CalculateSubDomain(unsigned int index) const;

		static std::array<float, 32> ShapeFunction(const glm::vec3& xi, std::array<std::array<float, 3>, 32>* gradient = nullptr);
	public:
		BoundingBox<glm::vec3> m_Domain;

		size_t m_CellCount;
		size_t m_FieldCount;

		glm::uvec3 m_Resolution;
		glm::vec3 m_CellSize;
		glm::vec3 m_CellSizeInverse;

		std::vector<std::vector<float>> m_Nodes;
		std::vector<std::vector<std::array<unsigned int, 32>>> m_Cells;
		std::vector<std::vector<unsigned int>> m_CellMap;
	};
}

#endif // !SDF_H