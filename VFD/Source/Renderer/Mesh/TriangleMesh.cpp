#include "pch.h"
#include "TriangleMesh.h"

#include "Utility/FileSystem.h"
#include "tiny_obj_loader.h"

namespace vfd {
	TriangleMesh::TriangleMesh(const std::string& filepath)
	{
		LoadOBJ(filepath);
	}

	TriangleMesh::TriangleMesh(const std::string& filepath, glm::vec3 scale)
	{
		LoadOBJ(filepath, scale);
	}

	TriangleMesh::TriangleMesh(const AABB& bbox)
	{
		glm::vec3 p = bbox.position;
		m_Vertices = {
		   glm::vec3(p.x, p.y, p.z),
		   glm::vec3(p.x + bbox.width, p.y, p.z),
		   glm::vec3(p.x + bbox.width, p.y, p.z + bbox.depth),
		   glm::vec3(p.x, p.y, p.z + bbox.depth),
		   glm::vec3(p.x, p.y + bbox.height, p.z),
		   glm::vec3(p.x + bbox.width, p.y + bbox.height, p.z),
		   glm::vec3(p.x + bbox.width, p.y + bbox.height, p.z + bbox.depth),
		   glm::vec3(p.x, p.y + bbox.height, p.z + bbox.depth)
		};

		m_Triangles = {
			 {0, 1, 2}, {0, 2, 3}, {4, 7, 6}, {4, 6, 5},
			 {0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2},
			 {0, 4, 5}, {0, 5, 1}, {3, 2, 6}, {3, 6, 7}
		};
	}

	void TriangleMesh::LoadOBJ(const std::string& filepath, glm::vec3 scale)
	{
		ASSERT(fs::FileExists(filepath), "filepath invalid (" + filepath + ")!")

		m_Filename = fs::FilenameFromFilepath(filepath);
		m_Filepath = filepath;

		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		tinyobj::attrib_t attributes;
		std::vector<float> buffer;

		std::string warning;
		std::string error;

		if (tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error, filepath.c_str()) == false) {
			if (error.empty() == false) {
				ERR(error, "triangle mesh");
			}
		}

		for (size_t i = 0; i < shapes.size(); i++) {
			for (size_t j = 0; j < shapes[i].mesh.indices.size() / 3; j++) {
				tinyobj::index_t index0 = shapes[i].mesh.indices[3 * j + 0];
				tinyobj::index_t index1 = shapes[i].mesh.indices[3 * j + 1];
				tinyobj::index_t index2 = shapes[i].mesh.indices[3 * j + 2];

				// Vertices + Triangles
				float v[3][3];
				glm::uvec3 triangle;
				for (int k = 0; k < 3; k++) {
					triangle.x = index0.vertex_index;
					triangle.y = index1.vertex_index;
					triangle.z = index2.vertex_index;

					v[0][k] = attributes.vertices[3 * triangle.x + k];
					v[1][k] = attributes.vertices[3 * triangle.y + k];
					v[2][k] = attributes.vertices[3 * triangle.z + k];
				}

				m_Triangles.push_back(triangle);

				// Normals
				float n[3][3];
				int nf0 = index0.normal_index;
				int nf1 = index1.normal_index;
				int nf2 = index2.normal_index;

				for (int k = 0; k < 3; k++) {
					n[0][k] = attributes.normals[3 * nf0 + k];
					n[1][k] = attributes.normals[3 * nf1 + k];
					n[2][k] = attributes.normals[3 * nf2 + k];
				}

				// Move data into a float buffer
				for (int k = 0; k < 3; k++) {
					// Vertices
					buffer.push_back(v[k][0] * scale.x);
					buffer.push_back(v[k][1] * scale.y);
					buffer.push_back(v[k][2] * scale.z);

					// Normals
					buffer.push_back(n[k][0]);
					buffer.push_back(n[k][1]);
					buffer.push_back(n[k][2]);

					m_Normals.push_back({n[k][0], n[k][1], n[k][2]});
				}
			}
		}

		for (size_t i = 0; i < attributes.vertices.size(); i += 3) {
			m_Vertices.push_back(glm::vec3(
				attributes.vertices[i + 0],
				attributes.vertices[i + 1],
				attributes.vertices[i + 2]
			) * scale);
		}

		m_VBO = Ref<VertexBuffer>::Create(buffer);
		m_VBO->SetLayout({
			{ShaderDataType::Float3, "a_Position"},
			{ShaderDataType::Float3, "a_Normal"}
		});

		m_VAO = Ref<VertexArray>::Create();
		m_VAO->AddVertexBuffer(m_VBO);
	}

	void TriangleMesh::Translate(const glm::vec3& value)
	{
		for (size_t i = 0; i < m_Vertices.size(); i++)
		{
			m_Vertices[i] += value;
		}
	}

	void TriangleMesh::Transform(const glm::mat4& value)
	{
		for (size_t i = 0; i < m_Vertices.size(); i++)
		{
			m_Vertices[i] =  static_cast<glm::vec3>(glm::vec4(m_Vertices[i], 0.0f) * value);
		}
	}

	const Ref<VertexArray>& TriangleMesh::GetVAO() const
	{
		return m_VAO;
	}

	uint32_t TriangleMesh::GetVertexCount() const
	{
		return static_cast<uint32_t>(m_Triangles.size() * 3);
	}

	uint32_t TriangleMesh::GetTriangleCount() const
	{
		return static_cast<uint32_t>(m_Triangles.size());
	}

	const std::vector<glm::vec3>& TriangleMesh::GetVertices()
	{
		return m_Vertices;
	}

	const std::vector<glm::uvec3>& TriangleMesh::GetTriangles()
	{
		return m_Triangles;
	}

	const std::vector<glm::vec3> TriangleMesh::CopyVertices() const
	{
		return m_Vertices;
	}

	const std::vector<glm::uvec3> TriangleMesh::CopyTriangles() const
	{
		return m_Triangles;
	}

	const std::string& TriangleMesh::GetSourceFilepath() const
	{
		return m_Filepath;
	}
}