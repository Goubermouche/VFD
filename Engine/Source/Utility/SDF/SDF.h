#ifndef SDF_H
#define SDF_H

#include "Core/Structures/BoundingBox.h"
#include "Renderer/Mesh/EdgeMesh.h"

namespace fe {
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

		SDF(const BoundingBox& domain, glm::ivec3 resolution);
		SDF(const EdgeMesh& mesh, const BoundingBox& bounds, const glm::uvec3& resolution, bool inverted = false);
		SDF(const std::string& filepath);
		~SDF() = default;

		float GetDistance(const glm::vec3& point, float thickness) const;
		uint32_t AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate = nullptr);

		float Interpolate(unsigned int fieldID, const glm::vec3& point, glm::vec3* gradient = nullptr) const;
		double Interpolate(unsigned int fieldID, const glm::dvec3& xi, const std::array<unsigned int, 32>& cell, const glm::dvec3& c0, const std::array<double, 32>& N,
			glm::dvec3* gradient = nullptr, std::array<std::array<double, 3>, 32>* dN = nullptr);

		const BoundingBox& GetDomain() const { return m_Domain; }
		bool DetermineShapeFunctions(unsigned int fieldID, const glm::dvec3& x, std::array<unsigned int, 32>& cell,
			glm::dvec3& c0, std::array<double, 32>& N, std::array<std::array<double, 3>, 32>* dN = nullptr);
	private:
		glm::vec3 IndexToNodePosition(uint32_t index) const;

		glm::ivec3 SingleToMultiIndex(uint32_t index) const;
		uint32_t MultiToSingleIndex(const glm::ivec3& index) const;

		BoundingBox CalculateSubDomain(const glm::vec3& index) const;
		BoundingBox CalculateSubDomain(uint32_t index) const;

		static std::array<double, 32> ShapeFunction(const glm::vec3& xi, std::array<std::array<double, 3>, 32>* gradient = nullptr);
	private:
		BoundingBox m_Domain;

		size_t m_CellCount;
		size_t m_FieldCount;

		glm::uvec3 m_Resolution;
		glm::dvec3 m_CellSize;
		glm::dvec3 m_CellSizeInverse;

		std::vector<std::vector<double>> m_Nodes;
		std::vector<std::vector<std::array<unsigned int, 32>>> m_Cells;
		std::vector<std::vector<unsigned int>> m_CellMap;
	};
}

#endif // !SDF_H