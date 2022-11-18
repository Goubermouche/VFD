#include "pch.h"
#include "BoundingSphere.h"

namespace vfd {
	MeshBoundingSphereHierarchy::MeshBoundingSphereHierarchy(const std::vector<glm::dvec3>& vertices, const std::vector<glm::ivec3>& faces)
		: Tree<BoundingSphere>(faces.size()), m_Vertices(vertices), m_Faces(faces), m_TriangleCenters(faces.size())
	{
		std::ranges::transform(m_Faces.begin(), m_Faces.end(), m_TriangleCenters.begin(),
		[&](const glm::ivec3& f)
		{
			return 1.0 / 3.0 * (m_Vertices[f[0]] + m_Vertices[f[1]] + m_Vertices[f[2]]);
		});
	}

	const glm::dvec3& MeshBoundingSphereHierarchy::GetEntityPosition(unsigned int index) const
	{
		return m_TriangleCenters[index];
	}

	void MeshBoundingSphereHierarchy::Calculate(const unsigned int b, const unsigned int n, BoundingSphere& hull) const
	{
		std::vector<glm::dvec3> vertexSubset = std::vector<glm::dvec3>(3 * n);
		for (unsigned int i(0); i < n; ++i)
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