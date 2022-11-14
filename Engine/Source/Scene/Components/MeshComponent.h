#ifndef MESH_COMPONENT_H
#define MESH_COMPONENT_H

#include "Renderer/Mesh/TriangleMesh.h"

namespace fe {
	class Application;

	struct MeshComponent {
		Ref<TriangleMesh> Mesh;

		MeshComponent() = default;
		MeshComponent(const MeshComponent& other) = default;
		MeshComponent(const std::string& filepath);

		template<class Archive>
		void save(Archive& archive) const;

		template<class Archive>
		void load(Archive& archive);
	};

	template<class Archive>
	inline void MeshComponent::save(Archive& archive) const
	{
		archive(cereal::make_nvp("meshSource", Mesh->GetSourceFilepath()));
	}

	template<class Archive>
	inline void MeshComponent::load(Archive& archive)
	{
		std::string meshSource;
		archive(cereal::make_nvp("meshSource", meshSource));

		Mesh = Ref<TriangleMesh>::Create(meshSource);
	}
}

#endif // !MESH_COMPONENT_H
