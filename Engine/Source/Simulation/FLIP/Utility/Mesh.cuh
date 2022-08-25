#ifndef MESH_CUH
#define MESH_CUH

#include "tiny_obj_loader.h"

namespace fe {
	struct Mesh {
		__device__ void Init(const char* filepath) {
			std::vector<tinyobj::shape_t> shapes;
			std::vector<tinyobj::material_t> materials;
			tinyobj::attrib_t attributes;
			std::vector<float> buffer;
		}

		glm::vec3* Vertices;
		int VertexCount;
	};
}
#endif // !MESH_CUH
