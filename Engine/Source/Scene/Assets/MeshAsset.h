#ifndef MESH_ASSET_H
#define MESH_ASSET_H

#include "Renderer/Mesh/TriangleMesh.h"

namespace fe {
	class MeshAsset : public Asset {
	public:
		MeshAsset(const std::string& filepath)
			: Asset(filepath)
		{
			m_Mesh = Ref<TriangleMesh>::Create(filepath);
		}

		Ref<TriangleMesh> GetMesh() const {
			return m_Mesh;
		}
	private:
		Ref<TriangleMesh> m_Mesh;
	};
}

#endif // !MESH_ASSET_H