#ifndef MATERIAL_COMPONENT_H_
#define MATERIAL_COMPONENT_H_

namespace fe {
	struct MaterialComponent {
		Ref<Material> MaterialHandle;

		MaterialComponent() = default;
		MaterialComponent(const MaterialComponent& other) = default;
		MaterialComponent(Ref<Material> material)
			: MaterialHandle(material) 
		{}
		MaterialComponent(const std::string& filePath) // shader filepath
			: MaterialHandle(Ref<Material>::Create(Ref<Shader>::Create(filePath)))
		{}

		template<class Archive>
		void save(Archive& archive) const
		{
			archive(cereal::make_nvp("shaderSource", MaterialHandle->GetShader()->GetSourceFilepath()));
		}

		template<class Archive>
		void load(Archive& archive)
		{
			std::string shaderSource;
			archive(cereal::make_nvp("shaderSource", shaderSource));
			MaterialHandle = Ref<Material>::Create(Ref<Shader>::Create(shaderSource));
			MaterialHandle->Set("color", { 0.4f, 1.0f, 1.0f }); // TEMP
			MaterialHandle->Set("colorSecondary", { 1.0f, 0.4f, 1.0f, 1.0f });
			// TODO: implement uniform serialization 
		}
	};
}

#endif // !MATERIAL_COMPONENT_H_