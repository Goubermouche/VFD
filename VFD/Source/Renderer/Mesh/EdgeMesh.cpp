#include "pch.h"
#include "EdgeMesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace vfd {
	template <class T>
	void HashCombine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	struct HalfEdgeHasher
	{
		HalfEdgeHasher(const std::vector<glm::uvec3>& faces)
			: Faces(&faces)
		{}

		std::size_t operator()(const HalfEdge& halfEdge) const
		{
			const unsigned int f = halfEdge.GetFace();
			const unsigned int e = halfEdge.GetEdge();
			std::array<unsigned int, 2> v = { (*Faces)[f][e], (*Faces)[f][(e + 1u) % 3u] };

			if (v[0] > v[1]) {
				std::swap(v[0], v[1]);
			}

			std::size_t seed(0);
			HashCombine(seed, v[0]);
			HashCombine(seed, v[1]);
			return seed;
		}

		const std::vector<glm::uvec3>* Faces;
	};

	struct HalfEdgeEqualTo
	{
		HalfEdgeEqualTo(const std::vector<glm::uvec3>& faces)
			: Faces(&faces)
		{}

		bool operator()(const HalfEdge& a, const HalfEdge& b) const
		{
			const unsigned int fa = a.GetFace();
			const unsigned int ea = a.GetEdge();
			const std::array<unsigned int, 2> va = { (*Faces)[fa][ea], (*Faces)[fa][(ea + 1) % 3] };

			const unsigned int fb = b.GetFace();
			const unsigned int eb = b.GetEdge();
			const std::array<unsigned int, 2> vb = { (*Faces)[fb][eb], (*Faces)[fb][(eb + 1) % 3] };

			return va[0] == vb[1] && va[1] == vb[0];
		}

		const std::vector<glm::uvec3>* Faces;
	};

	EdgeMesh::EdgeMesh(const std::string& filepath, const glm::vec3 Scale)
	{
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		tinyobj::attrib_t attributes;
		std::vector<unsigned int> faces;

		std::string warning;
		std::string error;

		if (tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error, filepath.c_str()) == false) {
			if (error.empty() == false) {
				ERR(error, "edge mesh")
			}
		}

		for (size_t j = 0; j < attributes.vertices.size(); j += 3)
		{
			m_Vertices.emplace_back(glm::vec3(
				attributes.vertices[j + 0] * Scale.x,
				attributes.vertices[j + 1] * Scale.y,
				attributes.vertices[j + 2] * Scale.z
			));
		}

		size_t indexOffset = 0;
		for (size_t f = 0; f < shapes[0].mesh.num_face_vertices.size(); f++) {
			for (size_t v = 0; v < 3; v++) {
				faces.push_back(shapes[0].mesh.indices[indexOffset + v].vertex_index);
			}

			indexOffset += 3;
		}
		
		m_Faces.resize(faces.size() / 3);
		std::copy_n(faces.data(), faces.size(), &(m_Faces.front().x));

		m_Edges.resize(3 * m_Faces.size());
		m_IncidentEdges.resize(m_Vertices.size());

		// Build adjacencies for mesh faces
		std::unordered_set<HalfEdge, HalfEdgeHasher, HalfEdgeEqualTo> faceSet((m_Faces.size() * 3) / 2, HalfEdgeHasher(m_Faces), HalfEdgeEqualTo(m_Faces));
		for (unsigned int i(0); i < m_Faces.size(); ++i) {
			for (unsigned char j(0); j < 3; ++j)
			{
				HalfEdge halfEdge(i, j);
				auto ret = faceSet.insert(halfEdge);
				if (!ret.second)
				{
					m_Edges[halfEdge.GetFace()][halfEdge.GetEdge()] = *(ret.first);
					m_Edges[ret.first->GetFace()][ret.first->GetEdge()] = halfEdge;

					faceSet.erase(ret.first);
				}

				m_IncidentEdges[m_Faces[i][j]] = halfEdge;
			}
		}

		m_BorderEdges.reserve(faceSet.size());

		for (HalfEdge const halfEdge : faceSet)
		{
			m_BorderEdges.push_back(halfEdge);
			HalfEdge b(static_cast<unsigned int>(m_BorderEdges.size()) - 1u, 3);
			m_Edges[halfEdge.GetFace()][halfEdge.GetEdge()] = b;
			m_IncidentEdges[Target(halfEdge)] = b;
		}
	}

	EdgeMesh::EdgeMesh(const std::vector<glm::vec3>& vertices, const std::vector<glm::uvec3>& faces)
		: m_Vertices(vertices), m_Faces(faces)
	{
		m_Edges.resize(3 * m_Faces.size());
		m_IncidentEdges.resize(m_Vertices.size());

		// Build adjacencies for mesh faces
		std::unordered_set<HalfEdge, HalfEdgeHasher, HalfEdgeEqualTo> faceSet((m_Faces.size() * 3) / 2, HalfEdgeHasher(m_Faces), HalfEdgeEqualTo(m_Faces));
		for (unsigned int i(0); i < m_Faces.size(); ++i) {
			for (unsigned char j(0); j < 3; ++j)
			{
				HalfEdge halfEdge(i, j);
				auto ret = faceSet.insert(halfEdge);
				if (!ret.second)
				{
					m_Edges[halfEdge.GetFace()][halfEdge.GetEdge()] = *(ret.first);
					m_Edges[ret.first->GetFace()][ret.first->GetEdge()] = halfEdge;

					faceSet.erase(ret.first);
				}

				m_IncidentEdges[m_Faces[i][j]] = halfEdge;
			}
		}

		m_BorderEdges.reserve(faceSet.size());

		for (HalfEdge const halfEdge : faceSet)
		{
			m_BorderEdges.push_back(halfEdge);
			HalfEdge b(static_cast<unsigned int>(m_BorderEdges.size()) - 1u, 3);
			m_Edges[halfEdge.GetFace()][halfEdge.GetEdge()] = b;
			m_IncidentEdges[Target(halfEdge)] = b;
		}
	}

	EdgeMesh& EdgeMesh::operator=(const EdgeMesh other)
	{
		m_Vertices = other.m_Vertices;
		m_Faces = other.m_Faces;
		m_Edges = other.m_Edges;
		m_IncidentEdges = other.m_IncidentEdges;
		m_BorderEdges = other.m_BorderEdges;

		WARN("MESH_DISTANCE: = OPERATOR CALLED")

		return*this;
	}


	FaceIterator FaceContainer::end() const
	{
		return FaceIterator(static_cast<unsigned int>(m_Mesh->GetFaceCount()), m_Mesh);
	}

	unsigned int FaceIterator::GetVertex(const unsigned int i ) const
	{
		return m_Mesh->GetFaceVertex(m_Index, i);
	}

	FaceIterator::reference	FaceIterator::operator*() const
	{
		return m_Mesh->GetFace(m_Index);
	}

	FaceConstIterator::reference FaceConstIterator::operator*() const
	{
		return m_Mesh->GetFace(m_Index);
	}

	FaceConstIterator FaceConstContainer::end() const
	{
		return FaceConstIterator(static_cast<unsigned int>(m_Mesh->GetFaceCount()), m_Mesh);
	}

	IncidentFaceIterator& IncidentFaceIterator::operator++()
	{
		const HalfEdge oppositeHalfEdge = m_Mesh->Opposite(m_HalfEdge);
		if (oppositeHalfEdge.IsBoundary())
		{
			m_HalfEdge = HalfEdge();
			return *this;
		}

		m_HalfEdge = oppositeHalfEdge.GetNext();
		if (m_HalfEdge == m_BeginHalfEdge)
		{
			m_HalfEdge = HalfEdge();
		}
		return *this;
	}

	IncidentFaceIterator::IncidentFaceIterator(const unsigned int v, const EdgeMesh* mesh)
		:  m_HalfEdge(mesh->GetIncidentHalfEdge(v)), m_BeginHalfEdge(mesh->GetIncidentHalfEdge(v)), m_Mesh(mesh)
	{
		if (m_HalfEdge.IsBoundary()) {
			m_HalfEdge = mesh->Opposite(m_HalfEdge).GetNext();
		}
	}
}
