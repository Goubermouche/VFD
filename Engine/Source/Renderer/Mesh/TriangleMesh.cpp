#include "pch.h"
#include "TriangleMesh.h"

#include "Utility/FileSystem.h"
#include "tiny_obj_loader.h"

namespace fe {
	TriangleMesh::TriangleMesh(const std::string& filepath)
	{
		ASSERT(FileExists(filepath), "filepath invalid (" + filepath + ")!");

		m_Filename = FilenameFromFilepath(filepath);
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

		ERR(shapes[0].mesh.num_face_vertices.size());
		ERR(attributes.vertices.size() / 3);

		for (size_t i = 0; i < shapes.size(); i++) {
			for (size_t j = 0; j < shapes[i].mesh.indices.size() / 3; j++) {
				tinyobj::index_t index0 = shapes[i].mesh.indices[3 * j + 0];
				tinyobj::index_t index1 = shapes[i].mesh.indices[3 * j + 1];
				tinyobj::index_t index2 = shapes[i].mesh.indices[3 * j + 2];

				// Vertices + Triangles
				float v[3][3];
				glm::ivec3 triangle;
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
					buffer.push_back(v[k][0]);
					buffer.push_back(v[k][1]);
					buffer.push_back(v[k][2]);

					m_Vertices.push_back({
						v[k][0],
						v[k][1],
						v[k][2],
						});

					// Normals
					buffer.push_back(n[k][0]);
					buffer.push_back(n[k][1]);
					buffer.push_back(n[k][2]);
				}
			}
		}

		Ref<VertexBuffer> vbo = Ref<VertexBuffer>::Create(buffer);
		vbo->SetLayout({
			{ShaderDataType::Float3, "a_Position"},
			{ShaderDataType::Float3, "a_Normal"}
		});

		m_VAO = Ref<VertexArray>::Create();
		m_VAO->AddVertexBuffer(vbo);

		LOG("mesh loaded (" + filepath + ")");
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
}