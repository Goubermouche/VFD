#ifndef EDGE_MESH_H
#define EDGE_MESH_H

#include "Core/Structures/BoundingSphere.h"

namespace vfd {
	enum class Triangle
	{
		A,
		B,
		C,
		D,
		E,
		F,
		G
	};

	class HalfEdge
	{
	public:
		HalfEdge()
			: m_Code(3)
		{}

		HalfEdge(HalfEdge const&) = default;
		HalfEdge(const unsigned int f,const unsigned char e)
			: m_Code((f << 2) | e)
		{}

		~HalfEdge() = default;

		HalfEdge GetNext() const
		{
			return HalfEdge(GetFace(), (GetEdge() + 1u) % 3u);
		}

		HalfEdge GetPrevious() const
		{
			return HalfEdge(GetFace(), (GetEdge() + 2u) % 3u);
		}

		bool operator==(HalfEdge const& other) const
		{
			return m_Code == other.m_Code;
		}

		[[nodiscard]]
		unsigned int GetFace() const {
			return m_Code >> 2;
		}

		[[nodiscard]]
		unsigned char GetEdge() const {
			return m_Code & 0x3u;
		}

		[[nodiscard]]
		bool IsBoundary() const {
			return GetEdge() == 3u;
		}
	private:
		HalfEdge(const unsigned int code)
			: m_Code(code)
		{}
	private:
		unsigned int m_Code = 0u;
	};

	class EdgeMesh;

	class FaceContainer;
	class FaceIterator : public	std::iterator<std::random_access_iterator_tag, glm::uvec3>
	{
	public:
		FaceIterator() = delete;
		~FaceIterator() = default;

		reference operator*() const;

		bool operator<(const FaceIterator& other) const
		{
			return m_Index < other.m_Index;
		}

		bool operator==(const FaceIterator& other) const
		{
			return m_Index == other.m_Index;
		}

		bool operator!=(const FaceIterator& other) const
		{
			return !(*this == other);
		}

		FaceIterator& operator++() {
			++m_Index; return *this;
		}

		FaceIterator& operator--() {
			--m_Index; return *this;
		}

		FaceIterator operator+(const FaceIterator& rhs) const
		{
			return FaceIterator(m_Index + rhs.m_Index, m_Mesh);
		}

		difference_type operator-(const FaceIterator& rhs) const 
		{
			return m_Index - rhs.m_Index;
		}

		FaceIterator operator-(const int& rhs) const
		{
			return FaceIterator(m_Index - rhs, m_Mesh);
		}

		[[nodiscard]]
		unsigned int GetVertex(unsigned int i) const;
	private:
		FaceIterator(const unsigned int index, EdgeMesh* mesh)
			: m_Index(index), m_Mesh(mesh)
		{}
	private:
		unsigned int m_Index = 0u;
		EdgeMesh* m_Mesh;

		friend class FaceContainer;
	};
	class FaceContainer
	{
	public:
		FaceContainer() = default;
		~FaceContainer() = default;

		[[nodiscard]]
		FaceIterator begin() const
		{
			return FaceIterator(0u, m_Mesh);
		}

		[[nodiscard]]
		FaceIterator end() const;
	private:
		FaceContainer(EdgeMesh* mesh)
			: m_Mesh(mesh)
		{}
	private:
		EdgeMesh* m_Mesh;

		friend class EdgeMesh;
	};

	class FaceConstIterator : public std::iterator<std::random_access_iterator_tag, glm::uvec3 const>
	{
	public:
		FaceConstIterator() = delete;
		~FaceConstIterator() = default;

		reference operator*() const;

		bool operator<(const FaceConstIterator& other) const
		{
			return m_Index < other.m_Index;
		}

		bool operator==(const FaceConstIterator& other) const
		{
			return m_Index == other.m_Index;
		}

		bool operator!=(const FaceConstIterator& other) const
		{
			return !(*this == other);
		}

		FaceConstIterator& operator++() {
			++m_Index; return *this;
		}

		FaceConstIterator& operator--() {
			--m_Index; return *this;
		}

		FaceConstIterator operator+(const FaceConstIterator& rhs) const
		{
			return FaceConstIterator(m_Index + rhs.m_Index, m_Mesh);
		}

		difference_type operator-(const FaceConstIterator& rhs) const
		{
			return m_Index - rhs.m_Index;
		}

		FaceConstIterator operator-(const int& rhs) const
		{
			return FaceConstIterator(m_Index - rhs, m_Mesh);
		}
	private:
		FaceConstIterator(const unsigned int index, const EdgeMesh* mesh)
			: m_Index(index), m_Mesh(mesh)
		{}
	private:
		unsigned int m_Index;
		const EdgeMesh* m_Mesh;

		friend class FaceConstContainer;
	};
	class FaceConstContainer
	{
	public:
		FaceConstContainer() = default;
		~FaceConstContainer() = default;

		[[nodiscard]]
		FaceConstIterator begin() const
		{
			return FaceConstIterator(0u, m_Mesh);
		}

		[[nodiscard]]
		FaceConstIterator end() const;
	private:
		FaceConstContainer(const EdgeMesh* mesh)
			: m_Mesh(mesh)
		{}
	private:
		const EdgeMesh* m_Mesh;

		friend class EdgeMesh;
	};

	class IncidentFaceContainer;
	class IncidentFaceIterator : public std::iterator<std::forward_iterator_tag, HalfEdge>
	{
	public:
		~IncidentFaceIterator() = default;

		value_type operator*() const
		{
			return m_HalfEdge;
		}

		IncidentFaceIterator& operator++();

		bool operator==(const IncidentFaceIterator& other) const
		{
			return m_HalfEdge == other.m_HalfEdge;
		}

		bool operator!=(const IncidentFaceIterator& other) const
		{
			return !(*this == other);
		}
	private:
		IncidentFaceIterator(unsigned int v, const EdgeMesh* mesh);
		IncidentFaceIterator()
			: m_Mesh(nullptr)
		{}
	private:
		HalfEdge m_HalfEdge, m_BeginHalfEdge;
		const EdgeMesh* m_Mesh;

		friend class IncidentFaceContainer;
	};

	class IncidentFaceContainer
	{
	public:
		IncidentFaceContainer() = default;
		~IncidentFaceContainer() = default;

		[[nodiscard]]
		IncidentFaceIterator begin() const
		{
			return IncidentFaceIterator(m_Vertex, m_Mesh);
		}

		[[nodiscard]]
		IncidentFaceIterator end() const
		{
			return IncidentFaceIterator();
		}
	private:
		IncidentFaceContainer(const unsigned int v, const EdgeMesh* mesh)
			: m_Mesh(mesh), m_Vertex(v)
		{}
	private:
		const EdgeMesh* m_Mesh;
		unsigned int m_Vertex = 0u;

		friend class EdgeMesh;
	};

	class EdgeMesh : public RefCounted {
	public:
		EdgeMesh(const std::string& filepath, glm::vec3 Scale = { 1.0f, 1.0f, 1.0f });
		EdgeMesh(const std::vector<glm::vec3>& vertices,const std::vector<glm::uvec3>& faces);
		~EdgeMesh() = default;

		/// <summary>
		/// Returns the origin of the specified halfedge.
		/// </summary>
		/// <param name="h">Target halfedge,</param>
		/// <returns>Index of the origin halfedge.</returns>
		[[nodiscard]]
		unsigned int Source(const HalfEdge  h) const
		{
			if (h.IsBoundary()) {
				return Target(Opposite(h));
			}

			return m_Faces[h.GetFace()][h.GetEdge()];
		}

		/// <summary>
		/// Returns the target halfedge of the specified halfedge.
		/// </summary>
		/// <param name="h">HalfEdge to find the target of.</param>
		/// <returns>Index of the target halfedge.</returns>
		[[nodiscard]]
		unsigned int Target(const HalfEdge  h) const
		{
			if (h.IsBoundary()) {
				return Source(Opposite(h));
			}

			return Source(h.GetNext());
		}

		/// <summary>
		/// Returns the specified halfedge's opposite.
		/// </summary>
		/// <param name="h">HalfEdge to find the opposite of.</param>
		/// <returns>Opposite halfedge.</returns>
		[[nodiscard]]
		HalfEdge Opposite(const HalfEdge  h) const
		{
			if (h.IsBoundary()) {
				return m_BorderEdges[h.GetFace()];
			}

			return m_Edges[h.GetFace()][h.GetEdge()];
		}

		[[nodiscard]]
		IncidentFaceContainer GetIncidentFaces(uint32_t v) const {
			return IncidentFaceContainer(v, this);
		}

		[[nodiscard]]
		std::size_t GetFaceCount() const {
			return m_Faces.size();
		}

		[[nodiscard]]
		std::size_t GetVertexCount() const {
			return m_IncidentEdges.size();
		}

		[[nodiscard]]
		std::size_t GetBorderEdgeCount() const {
			return m_BorderEdges.size();
		}

		[[nodiscard]]
		const std::vector<glm::vec3>& GetVertices() const {
			return m_Vertices;
		}

		std::vector<glm::vec3>& GetVertices() {
			return m_Vertices;
		}

		[[nodiscard]]
		const std::vector<glm::uvec3>& GetFaces() const {
			return m_Faces;
		}

		std::vector<glm::uvec3>& GetFaces() {
			return m_Faces;
		}

		[[nodiscard]]
		const glm::vec3& GetVertex(const unsigned int i) const {
			return m_Vertices[i];
		}

		glm::vec3& GetVertex(const unsigned int i) {
			return m_Vertices[i];
		}

		[[nodiscard]]
		const glm::uvec3& GetFace(const unsigned int i) const {
			return m_Faces[i];
		}

		glm::uvec3& GetFace(const unsigned int i) {
			return m_Faces[i];
		}

		[[nodiscard]]
		unsigned int const& GetFaceVertex(const unsigned int f, const unsigned int i) const
		{
			assert(i < 3u);
			assert(f < m_Faces.size());
			return m_Faces[f][i];
		}

		[[nodiscard]]
		HalfEdge GetIncidentHalfEdge(const unsigned int v) const {
			return m_IncidentEdges[v];
		}
	private:
		std::vector<glm::vec3> m_Vertices;
		std::vector<glm::uvec3> m_Faces;
		std::vector<std::array<HalfEdge, 3>> m_Edges;
		std::vector<HalfEdge> m_IncidentEdges;
		std::vector<HalfEdge> m_BorderEdges;
	};
}


#endif // !EDGE_MESH_H