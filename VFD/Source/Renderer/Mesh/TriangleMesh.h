#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "Renderer/VertexArray.h"
#include "Core/Structures/AxisAlignedBoundingBox.h"

namespace vfd {
	class TriangleMesh : public RefCounted	
	{
	public:
		TriangleMesh() = default;
		TriangleMesh(const std::string& filepath);
		TriangleMesh(const std::string& filepath, glm::vec3 scale);
		TriangleMesh(const AABB& aabb);
		~TriangleMesh() = default;

		void LoadOBJ(const std::string& filepath, glm::vec3 scale = { 1.0f, 1.0f, 1.0f });
		void Translate(const glm::vec3& value);

		const Ref<VertexArray>& GetVAO();

		uint32_t GetVertexCount() const;
		uint32_t GetTriangleCount() const;

		const std::vector<glm::vec3>& GetVertices();
		const std::vector<glm::uvec3>& GetTriangles();

		const std::vector<glm::vec3> CopyVertices() const;
		const std::vector<glm::uvec3> CopyTriangles() const;

		const std::string& GetSourceFilepath() const;
	private:
		Ref<VertexArray> m_VAO;

		std::string m_Filepath;
		std::string m_Filename;

		std::vector<glm::vec3> m_Vertices;
		std::vector<glm::uvec3> m_Triangles; // Indices to a vertex
	};
}

#endif // !TRIANGLE_MESH_H