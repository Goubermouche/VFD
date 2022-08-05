#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "Renderer/VertexArray.h"

namespace fe {
	class TriangleMesh : public RefCounted	
	{
	public:
		TriangleMesh(const std::string& filepath);
		~TriangleMesh() = default;

		const Ref<VertexArray>& GetVAO() {
			return m_VAO;
		}

		uint32_t GetVertexCount() const {
			return m_Vertices.size();
		}

		const std::vector<glm::vec3>& GetVertices() {
			return m_Vertices;
		}

		const std::string& GetSourceFilepath() const {
			return m_Filepath;
		}
	private:
		Ref<VertexArray> m_VAO;
		std::string m_Filepath;
		std::string m_Filename;

		std::vector<glm::vec3> m_Vertices;
	};
}

#endif // !TRIANGLE_MESH_H