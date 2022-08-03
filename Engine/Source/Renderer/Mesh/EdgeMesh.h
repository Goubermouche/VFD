#ifndef EDGE_MESH_H_
#define EDGE_MESH_H_


#include "Utility/String.h"
#include "Core/Structures/BoundingSphere.h"

namespace fe {
	enum class Triangle
	{
		VN0, VN1, VN2, EN0, EN1, EN2, FN
	};

	class Halfedge
	{
	public:
		Halfedge()
			: m_Code(3)
		{}
		Halfedge(Halfedge const&) = default;
		Halfedge(uint32_t f, unsigned char e)
			: m_Code((f << 2) | e)
		{}

		Halfedge GetNext() const
		{
			return Halfedge(GetFace(), (GetEdge() + 1) % 3);
		}

		Halfedge GetPrevious() const
		{
			return Halfedge(GetFace(), (GetEdge() + 2) % 3);
		}

		bool operator==(Halfedge const& other) const
		{
			return m_Code == other.m_Code;
		}

		uint32_t GetFace() const {
			return m_Code >> 2;
		}

		unsigned char GetEdge() const {
			return m_Code & 0x3;
		}

		bool IsBoundary() const {
			return GetEdge() == 3;
		}
	private:
		Halfedge(uint32_t code)
			: m_Code(code)
		{}
	private:
		uint32_t m_Code;
	};

	class EdgeMesh;

	class FaceContainer;
	class FaceIterator : public	std::iterator<std::random_access_iterator_tag, glm::ivec3>
	{
	public:
		FaceIterator() = delete;

		reference operator*();

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

		inline FaceIterator& operator++() {
			++m_Index; return *this;
		}

		inline FaceIterator& operator--() {
			--m_Index; return *this;
		}

		inline FaceIterator operator+(const FaceIterator& rhs)
		{
			return FaceIterator(m_Index + rhs.m_Index, m_Mesh);
		}

		inline difference_type operator-(const FaceIterator& rhs)
		{
			return m_Index - rhs.m_Index;
		}

		inline FaceIterator operator-(const int32_t& rhs)
		{
			return FaceIterator(m_Index - rhs, m_Mesh);
		}

		uint32_t GetVertex(uint32_t i) const;
	private:
		FaceIterator(uint32_t index, EdgeMesh* mesh)
			: m_Index(index), m_Mesh(mesh)
		{}
	private:
		uint32_t m_Index;
		EdgeMesh* m_Mesh;

		friend class FaceContainer;
	};
	class FaceContainer
	{
	public:
		FaceIterator begin() const
		{
			return FaceIterator(0, m_Mesh);
		}

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

		reference operator*();

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

		inline FaceConstIterator& operator++() {
			++m_Index; return *this;
		}

		inline FaceConstIterator& operator--() {
			--m_Index; return *this;
		}

		inline FaceConstIterator operator+(const FaceConstIterator& rhs) const
		{
			return FaceConstIterator(m_Index + rhs.m_Index, m_Mesh);
		}

		inline difference_type operator-(const FaceConstIterator& rhs) const
		{
			return m_Index - rhs.m_Index;
		}

		inline FaceConstIterator operator-(const int32_t& rhs) const
		{
			return FaceConstIterator(m_Index - rhs, m_Mesh);
		}
	private:
		FaceConstIterator(uint32_t index, const EdgeMesh* mesh)
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
		FaceConstIterator begin() const
		{
			return FaceConstIterator(0, m_Mesh);
		}
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
	class IncidentFaceIterator : public std::iterator<std::forward_iterator_tag, Halfedge>
	{
	public:
		value_type operator*() {
			return m_Halfedge;
		}

		IncidentFaceIterator& operator++();

		bool operator==(const IncidentFaceIterator& other) const
		{
			return m_Halfedge == other.m_Halfedge;
		}

		bool operator!=(const IncidentFaceIterator& other) const
		{
			return !(*this == other);
		}
	private:
		IncidentFaceIterator(uint32_t v, const EdgeMesh* mesh);
		IncidentFaceIterator()
			: m_Halfedge(), m_BeginHalfedge(), m_Mesh(nullptr)
		{}
	private:
		Halfedge m_Halfedge, m_BeginHalfedge;
		const EdgeMesh* m_Mesh;

		friend class IncidentFaceContainer;
	};

	class IncidentFaceContainer
	{
	public:
		IncidentFaceIterator begin() const
		{
			return IncidentFaceIterator(m_v, m_Mesh);
		}

		IncidentFaceIterator end() const
		{
			return IncidentFaceIterator();
		}
	private:
		IncidentFaceContainer(uint32_t v, const EdgeMesh* mesh)
			: m_v(v), m_Mesh(mesh)
		{}
	private:
		const EdgeMesh* m_Mesh;
		uint32_t m_v;

		friend class EdgeMesh;
	};

	class EdgeMesh {
	public:
		EdgeMesh(const std::string& filepath, const glm::vec3 scale = { 1, 1, 1 });
		EdgeMesh(const std::vector<glm::vec3>& vertices, const std::vector<uint32_t> faces);

		/// <summary>
		/// Gets the origin of the specifed halfedge.
		/// </summary>
		/// <param name="h">Target halfedge,</param>
		/// <returns>Index of the origin halfedge.</returns>
		uint32_t Source(const Halfedge  h) const
		{
			if (h.IsBoundary()) {
				return Target(Opposite(h));
			}

			return m_Faces[h.GetFace()][h.GetEdge()];
		}

		/// <summary>
		/// Gets the target halfedge of the specified halfedge.
		/// </summary>
		/// <param name="h">Halfedge to find the target of.</param>
		/// <returns>Index of the target halfedge.</returns>
		uint32_t Target(const Halfedge  h) const
		{
			if (h.IsBoundary()) {
				return Source(Opposite(h));
			}

			return Source(h.GetNext());
		}

		/// <summary>
		/// Gets the specified halfedge's opposite.
		/// </summary>
		/// <param name="h">Halfedge to find the opposite of.</param>
		/// <returns>Opposite halfedge.</returns>
		Halfedge Opposite(const Halfedge  h) const
		{
			if (h.IsBoundary()) {
				return m_BorderEdges[h.GetFace()];
			}

			return m_Edges[h.GetFace()][h.GetEdge()];
		}

		IncidentFaceContainer GetIncidentFaces(uint32_t v) const {
			return IncidentFaceContainer(v, this);
		}

		std::size_t GetFaceCount() const {
			return m_Faces.size();
		}

		std::size_t GetVertexCount() const {
			return m_IncidentEdges.size();
		}

		std::size_t GetBorderEdgeCount() const {
			return m_BorderEdges.size();
		}

		const std::vector<glm::vec3>& GetVertices() const {
			return m_Vertices;
		}

		std::vector<glm::vec3>& GetVertices() {
			return m_Vertices;
		}

		const std::vector<glm::ivec3>& GetFaces() const {
			return m_Faces;
		}

		std::vector<glm::ivec3>& GetFaces() {
			return m_Faces;
		}

		const glm::vec3& GetVertex(uint32_t i) const {
			return m_Vertices[i];
		}

		glm::vec3& GetVertex(uint32_t i) {
			return m_Vertices[i];
		}

		const glm::ivec3& GetFace(uint32_t i) const {
			return m_Faces[i];
		}

		glm::ivec3& GetFace(uint32_t i) {
			return m_Faces[i];
		}

		uint32_t const& GetFaceVertex(uint32_t f, uint32_t i) const
		{
			assert(i < 3);
			assert(f < m_Faces.size());
			return m_Faces[f][i];
		}

		Halfedge GetIncidentHalfedge(uint32_t v) const {
			return m_IncidentEdges[v];
		}
	private:
		std::vector<glm::vec3> m_Vertices;
		std::vector<glm::ivec3> m_Faces;

		std::vector<std::array<Halfedge, 3>> m_Edges;
		std::vector<Halfedge> m_IncidentEdges;
		std::vector<Halfedge> m_BorderEdges;
	};
}


#endif // !EDGE_MESH_H_
