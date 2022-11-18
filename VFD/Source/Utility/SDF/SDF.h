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
		using ContinuousFunction = std::function<float(const glm::dvec3&)>;
		using SamplePredicate = std::function<bool(const glm::dvec3&)>;

		SDF(const BoundingBox& domain, glm::ivec3 resolution);
		SDF(const EdgeMesh& mesh, const BoundingBox& bounds, const glm::uvec3& resolution, bool inverted = false);
		SDF(const std::string& filepath);
		~SDF() = default;

		double GetDistance(const glm::dvec3& point, double thickness) const;
		unsigned int AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate = nullptr);

		float Interpolate(unsigned int fieldID, const glm::dvec3& point, glm::dvec3* gradient = nullptr) const;
		double Interpolate(unsigned int fieldID, const glm::dvec3& xi, const std::array<unsigned int, 32>& cell, const glm::dvec3& c0, const std::array<double, 32>& N,
			glm::dvec3* gradient = nullptr, std::array<std::array<double, 3>, 32>* dN = nullptr);

		const BoundingBox& GetDomain() const { return m_Domain; }
		bool DetermineShapeFunctions(unsigned int fieldID, const glm::dvec3& x, std::array<unsigned int, 32>& cell,
			glm::dvec3& c0, std::array<double, 32>& N, std::array<std::array<double, 3>, 32>* dN = nullptr);
	private:
		glm::dvec3 IndexToNodePosition(unsigned int index) const;

		glm::uvec3 SingleToMultiIndex(unsigned int index) const;
		unsigned int MultiToSingleIndex(const glm::uvec3& index) const;

		BoundingBox CalculateSubDomain(const glm::uvec3& index) const;
		BoundingBox CalculateSubDomain(unsigned int index) const;

		static std::array<double, 32> ShapeFunction(const glm::dvec3& xi, std::array<std::array<double, 3>, 32>* gradient = nullptr);
	public:
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