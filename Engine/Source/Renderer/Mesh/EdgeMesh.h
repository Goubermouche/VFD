#ifndef EDGE_MESH_H
#define EDGE_MESH_H

#include "Core/Structures/BoundingSphere.h"

namespace fe {
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
		HalfEdge(const uint32_t f,const unsigned char e)
			: m_Code((f << 2) | e)
		{}

		~HalfEdge() = default;

		HalfEdge GetNext() const
		{
			return HalfEdge(GetFace(), (GetEdge() + 1) % 3);
		}

		HalfEdge GetPrevious() const
		{
			return HalfEdge(GetFace(), (GetEdge() + 2) % 3);
		}

		bool operator==(HalfEdge const& other) const
		{
			return m_Code == other.m_Code;
		}

		[[nodiscard]]
		uint32_t GetFace() const {
			return m_Code >> 2;
		}

		[[nodiscard]]
		unsigned char GetEdge() const {
			return m_Code & 0x3;
		}

		[[nodiscard]]
		bool IsBoundary() const {
			return GetEdge() == 3;
		}
	private:
		HalfEdge(const uint32_t code)
			: m_Code(code)
		{}
	private:
		uint32_t m_Code = 0;
	};

	class EdgeMesh;

	class FaceContainer;
	class FaceIterator : public	std::iterator<std::random_access_iterator_tag, glm::ivec3>
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

		FaceIterator operator-(const int32_t& rhs) const
		{
			return FaceIterator(m_Index - rhs, m_Mesh);
		}

		[[nodiscard]]
		uint32_t GetVertex(uint32_t i) const;
	private:
		FaceIterator(const uint32_t index, EdgeMesh* mesh)
			: m_Index(index), m_Mesh(mesh)
		{}
	private:
		uint32_t m_Index = 0;
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
			return FaceIterator(0, m_Mesh);
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

	class FaceConstIterator : public std::iterator<std::random_access_iterator_tag, glm::ivec3 const>
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

		FaceConstIterator operator-(const int32_t& rhs) const
		{
			return FaceConstIterator(m_Index - rhs, m_Mesh);
		}
	private:
		FaceConstIterator(const uint32_t index, const EdgeMesh* mesh)
			: m_Index(index), m_Mesh(mesh)
		{}
	private:
		uint32_t m_Index;
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
			return FaceConstIterator(0, m_Mesh);
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
		IncidentFaceIterator(uint32_t v, const EdgeMesh* mesh);
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
		IncidentFaceContainer(const uint32_t v, const EdgeMesh* mesh)
			: m_Mesh(mesh), m_Vertex(v)
		{}
	private:
		const EdgeMesh* m_Mesh;
		uint32_t m_Vertex = 0;

		friend class EdgeMesh;
	};

	class EdgeMesh {
	public:
		EdgeMesh(const std::string& filepath, glm::vec3 scale = { 1, 1, 1 });
		EdgeMesh(const std::vector<glm::vec3>& vertices,const std::vector<glm::ivec3>& faces);
		~EdgeMesh() = default;

		/// <summary>
		/// Gets the origin of the specified halfedge.
		/// </summary>
		/// <param name="h">Target halfedge,</param>
		/// <returns>Index of the origin halfedge.</returns>
		[[nodiscard]]
		uint32_t Source(const HalfEdge  h) const
		{
			if (h.IsBoundary()) {
				return Target(Opposite(h));
			}

			return m_Faces[h.GetFace()][h.GetEdge()];
		}

		/// <summary>
		/// Gets the target halfedge of the specified halfedge.
		/// </summary>
		/// <param name="h">HalfEdge to find the target of.</param>
		/// <returns>Index of the target halfedge.</returns>
		[[nodiscard]]
		uint32_t Target(const HalfEdge  h) const
		{
			if (h.IsBoundary()) {
				return Source(Opposite(h));
			}

			return Source(h.GetNext());
		}

		/// <summary>
		/// Gets the specified halfedge's opposite.
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
		const std::vector<glm::ivec3>& GetFaces() const {
			return m_Faces;
		}

		std::vector<glm::ivec3>& GetFaces() {
			return m_Faces;
		}

		[[nodiscard]]
		const glm::vec3& GetVertex(const uint32_t i) const {
			return m_Vertices[i];
		}

		glm::vec3& GetVertex(const uint32_t i) {
			return m_Vertices[i];
		}

		[[nodiscard]]
		const glm::ivec3& GetFace(const uint32_t i) const {
			return m_Faces[i];
		}

		glm::ivec3& GetFace(const uint32_t i) {
			return m_Faces[i];
		}

		[[nodiscard]]
		uint32_t const& GetFaceVertex(const uint32_t f, const uint32_t i) const
		{
			assert(i < 3);
			assert(f < m_Faces.size());
			return m_Faces[f][i];
		}

		[[nodiscard]]
		HalfEdge GetIncidentHalfEdge(const uint32_t v) const {
			return m_IncidentEdges[v];
		}
	private:
		std::vector<glm::vec3> m_Vertices;
		std::vector<glm::ivec3> m_Faces;
		std::vector<std::array<HalfEdge, 3>> m_Edges;
		std::vector<HalfEdge> m_IncidentEdges;
		std::vector<HalfEdge> m_BorderEdges;
	};
}


#endif // !EDGE_MESH_H