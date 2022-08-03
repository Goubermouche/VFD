#include "pch.h"
#include "TriangleMesh.h"

#include "Utility/FileSystem.h"
#include "tiny_obj_loader.h"

namespace fe {
	TriangleMesh::TriangleMesh(const std::string& filepath)
	{
		ASSERT(FileExists(filepath), "filepath invalid!");

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

		for (size_t i = 0; i < shapes.size(); i++) {
			for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
				tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * f + 0];
				tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * f + 1];
				tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * f + 2];

				// Vertices
				float v[3][3];
				for (int k = 0; k < 3; k++) {
					int f0 = idx0.vertex_index;
					int f1 = idx1.vertex_index;
					int f2 = idx2.vertex_index;

					v[0][k] = attributes.vertices[3 * f0 + k];
					v[1][k] = attributes.vertices[3 * f1 + k];
					v[2][k] = attributes.vertices[3 * f2 + k];
				}

				// Normals
				float n[3][3];
				int nf0 = idx0.normal_index;
				int nf1 = idx1.normal_index;
				int nf2 = idx2.normal_index;

				for (int k = 0; k < 3; k++) {
					n[0][k] = attributes.normals[3 * nf0 + k];
					n[1][k] = attributes.normals[3 * nf1 + k];
					n[2][k] = attributes.normals[3 * nf2 + k];
				}

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
	}
}