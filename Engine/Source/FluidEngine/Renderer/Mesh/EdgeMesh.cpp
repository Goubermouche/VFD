#include "pch.h"
#include "EdgeMesh.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation. Requires C++11
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

namespace fe {
	template <class T>
	inline void HashCombine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	struct HalfedgeHasher
	{
		HalfedgeHasher(const std::vector<glm::ivec3>& faces)
			: Faces(&faces)
		{}

		std::size_t operator()(const Halfedge& halfedge) const
		{
			uint32_t f = halfedge.GetFace();
			uint32_t e = halfedge.GetEdge();
			std::array<uint32_t, 2> v = { (*Faces)[f][e], (*Faces)[f][(e + 1) % 3] };

			if (v[0] > v[1]) {
				std::swap(v[0], v[1]);
			}

			std::size_t seed(0);
			HashCombine(seed, v[0]);
			HashCombine(seed, v[1]);
			return seed;
		}

		const std::vector<glm::ivec3>* Faces;
	};

	struct HalfedgeEqualTo
	{
		HalfedgeEqualTo(const std::vector<glm::ivec3>& faces)
			: Faces(&faces)
		{}

		bool operator()(const Halfedge& a, const Halfedge& b) const
		{
			uint32_t fa = a.GetFace();
			uint32_t ea = a.GetEdge();
			std::array<uint32_t, 2> va = { (*Faces)[fa][ea], (*Faces)[fa][(ea + 1) % 3] };

			uint32_t fb = b.GetFace();
			uint32_t eb = b.GetEdge();
			std::array<uint32_t, 2> vb = { (*Faces)[fb][eb], (*Faces)[fb][(eb + 1) % 3] };

			return va[0] == vb[1] && va[1] == vb[0];
		}

		const std::vector<glm::ivec3>* Faces;
	};

	EdgeMesh::EdgeMesh(const std::string& filepath, const glm::vec3 scale)
	{
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		tinyobj::attrib_t attributes;
		std::vector<uint32_t> faces;

		std::string warning;
		std::string error;

		if (tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error, filepath.c_str()) == false) {
			if (error.empty() == false) {
				ERR(error, "edge mesh");
			}
		}

		for (size_t j = 0; j < attributes.vertices.size(); j += 3)
		{
			m_Vertices.push_back(glm::vec3(
				attributes.vertices[j + 0] * scale.x,
				attributes.vertices[j + 1] * scale.y,
				attributes.vertices[j + 2] * scale.z
			));
		}

		size_t indexOffset = 0;
		for (size_t f = 0; f < shapes[0].mesh.num_face_vertices.size(); f++) {
			for (size_t v = 0; v < 3; v++) {
				tinyobj::index_t idx = shapes[0].mesh.indices[indexOffset + v];
				faces.push_back(idx.vertex_index);
			}

			indexOffset += 3;
		}
		
		m_Faces.resize(faces.size() / 3);
		std::copy_n(faces.data(), faces.size(), &(m_Faces.front().x));

		m_Edges.resize(3 * m_Faces.size());
		m_IncidentEdges.resize(m_Vertices.size());

		// Build adjacencies for mesh faces
		std::unordered_set<Halfedge, HalfedgeHasher, HalfedgeEqualTo> faceSet((m_Faces.size() * 3) / 2, HalfedgeHasher(m_Faces), HalfedgeEqualTo(m_Faces));
		for (uint32_t i(0); i < m_Faces.size(); ++i) {
			for (unsigned char j(0); j < 3; ++j)
			{
				Halfedge halfedge(i, j);
				auto ret = faceSet.insert(halfedge);
				if (!ret.second)
				{
					m_Edges[halfedge.GetFace()][halfedge.GetEdge()] = *(ret.first);
					m_Edges[ret.first->GetFace()][ret.first->GetEdge()] = halfedge;

					faceSet.erase(ret.first);
				}

				m_IncidentEdges[m_Faces[i][j]] = halfedge;
			}
		}

		m_BorderEdges.reserve(faceSet.size());

		for (Halfedge const halfedge : faceSet)
		{
			m_BorderEdges.push_back(halfedge);
			Halfedge b(static_cast<uint32_t>(m_BorderEdges.size()) - 1u, 3);
			m_Edges[halfedge.GetFace()][halfedge.GetEdge()] = b;
			m_IncidentEdges[Target(halfedge)] = b;

			assert(Source(b) == Target(halfedge));
		}
	}

	EdgeMesh::EdgeMesh(const std::vector< glm::vec3>& vertices, const std::vector<uint32_t> faces)
		: m_Vertices(vertices)
	{
		m_Faces.resize(faces.size() / 3);
		std::copy_n(faces.data(), faces.size(), &(m_Faces.front().x));

		m_Edges.resize(3 * m_Faces.size());
		m_IncidentEdges.resize(m_Vertices.size());

		// Build adjacencies for mesh faces
		std::unordered_set<Halfedge, HalfedgeHasher, HalfedgeEqualTo> faceSet((m_Faces.size() * 3) / 2, HalfedgeHasher(m_Faces), HalfedgeEqualTo(m_Faces));
		for (uint32_t i(0); i < m_Faces.size(); ++i) {
			for (unsigned char j(0); j < 3; ++j)
			{
				Halfedge halfedge(i, j);
				auto ret = faceSet.insert(halfedge);
				if (!ret.second)
				{
					m_Edges[halfedge.GetFace()][halfedge.GetEdge()] = *(ret.first);
					m_Edges[ret.first->GetFace()][ret.first->GetEdge()] = halfedge;

					faceSet.erase(ret.first);
				}

				m_IncidentEdges[m_Faces[i][j]] = halfedge;
			}
		}

		m_BorderEdges.reserve(faceSet.size());

		for (Halfedge const halfedge : faceSet)
		{
			m_BorderEdges.push_back(halfedge);
			Halfedge b(static_cast<uint32_t>(m_BorderEdges.size()) - 1u, 3);
			m_Edges[halfedge.GetFace()][halfedge.GetEdge()] = b;
			m_IncidentEdges[Target(halfedge)] = b;

			assert(Source(b) == Target(halfedge));
		}
	}


	FaceIterator FaceContainer::end() const
	{
		return FaceIterator(static_cast<uint32_t>(m_Mesh->GetFaceCount()), m_Mesh);
	}

	uint32_t FaceIterator::GetVertex(uint32_t i) const
	{
		return m_Mesh->GetFaceVertex(m_Index, i);
	}

	FaceIterator::reference	FaceIterator::operator*()
	{
		return m_Mesh->GetFace(m_Index);
	}

	FaceConstIterator::reference FaceConstIterator::operator*()
	{
		return m_Mesh->GetFace(m_Index);
	}

	FaceConstIterator FaceConstContainer::end() const
	{
		return FaceConstIterator(static_cast<uint32_t>(m_Mesh->GetFaceCount()), m_Mesh);
	}

	IncidentFaceIterator& IncidentFaceIterator::operator++()
	{
		Halfedge oppositeHalfedge = m_Mesh->Opposite(m_Halfedge);
		if (oppositeHalfedge.IsBoundary())
		{
			m_Halfedge = Halfedge();
			return *this;
		}

		m_Halfedge = oppositeHalfedge.GetNext();
		if (m_Halfedge == m_BeginHalfedge)
		{
			m_Halfedge = Halfedge();
		}
		return *this;
	}

	IncidentFaceIterator::IncidentFaceIterator(uint32_t v, const EdgeMesh* mesh)
		: m_Mesh(mesh), m_Halfedge(mesh->GetIncidentHalfedge(v)), m_BeginHalfedge(mesh->GetIncidentHalfedge(v))
	{
		if (m_Halfedge.IsBoundary()) {
			m_Halfedge = mesh->Opposite(m_Halfedge).GetNext();
		}
	}
}
