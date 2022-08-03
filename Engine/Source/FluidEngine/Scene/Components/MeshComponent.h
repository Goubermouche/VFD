#ifndef MESH_COMPONENT_H_
#define MESH_COMPONENT_H_

namespace fe {
	struct MeshComponent {
		Ref<TriangleMesh> Mesh;

		MeshComponent() = default;
		MeshComponent(const MeshComponent& other) = default;
		MeshComponent(const std::string& filePath)
			: Mesh(Ref<TriangleMesh>::Create(filePath))
		{}

		template<class Archive>
		void save(Archive& archive) const
		{
			archive(cereal::make_nvp("meshSource", Mesh->GetSourceFilepath()));
		}

		template<class Archive>
		void load(Archive& archive)
		{
			std::string meshSource;
			archive(cereal::make_nvp("meshSource", meshSource));

			Mesh = Ref<TriangleMesh>::Create(meshSource);
		}
	};
}

#endif // !MESH_COMPONENT_H_
