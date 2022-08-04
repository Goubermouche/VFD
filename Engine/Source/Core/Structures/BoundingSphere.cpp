#include "pch.h"
#include "BoundingSphere.h"

namespace fe {
	MeshBoundingSphereHierarchy::MeshBoundingSphereHierarchy(const std::vector<glm::vec3>& vertices, const std::vector<glm::ivec3>& faces)
		: Tree<BoundingSphere>(faces.size()), m_Vertices(vertices), m_Faces(faces), m_TriangleCenters(faces.size())
	{
		std::ranges::transform(m_Faces.begin(), m_Faces.end(), m_TriangleCenters.begin(),
		[&](const glm::ivec3& f)
		{
			return 1.0f / 3.0f * (m_Vertices[f[0]] + m_Vertices[f[1]] + m_Vertices[f[2]]);
		});
	}

	const glm::vec3& MeshBoundingSphereHierarchy::GetEntityPosition(uint32_t index) const
	{
		return m_TriangleCenters[index];
	}

	void MeshBoundingSphereHierarchy::Calculate(const uint32_t b, const uint32_t n, BoundingSphere& hull) const
	{
		std::vector<glm::vec3> vertexSubset = std::vector<glm::vec3>(3 * n);
		for (uint32_t i(0); i < n; ++i)
		{
			const auto& f = m_Faces[m_List[b + i]];
			{
				vertexSubset[3 * i + 0] = m_Vertices[f[0]];
				vertexSubset[3 * i + 1] = m_Vertices[f[1]];
				vertexSubset[3 * i + 2] = m_Vertices[f[2]];
			}
		}

		hull = BoundingSphere(vertexSubset);
	}
}