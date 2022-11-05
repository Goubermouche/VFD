#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "Renderer/VertexArray.h"
#include "Core/Structures/AxisAlignedBoundingBox.h"

namespace fe {
	class TriangleMesh : public RefCounted	
	{
	public:
		TriangleMesh() = default;
		TriangleMesh(const std::string& filepath);
		TriangleMesh(const std::string& filepath, glm::vec3 scale);
		TriangleMesh(const AABB& aabbs);
		~TriangleMesh() = default;

		void LoadOBJ(const std::string& filepath, glm::vec3 scale = {1, 1, 1});

		void Translate(const glm::vec3& value);

		const Ref<VertexArray>& GetVAO() {
			return m_VAO;
		}

		uint32_t GetVertexCount() {
			return m_Triangles.size() * 3;
		}

		uint32_t GetTriangleCount() const {
			return m_Triangles.size();
		}

		std::vector<glm::vec3>& GetVertices() {
			return m_Vertices;
		}

		std::vector<glm::ivec3>& GetTriangles() {
			return m_Triangles;
		}

		const std::string& GetSourceFilepath() const {
			return m_Filepath;
		}
	private:
		Ref<VertexArray> m_VAO;

		std::string m_Filepath;
		std::string m_Filename;

		std::vector<glm::vec3> m_Vertices;
		std::vector<glm::ivec3> m_Triangles; // Indices to a vertex
	};
}

#endif // !TRIANGLE_MESH_H