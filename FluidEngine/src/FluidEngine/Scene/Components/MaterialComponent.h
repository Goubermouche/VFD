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
			const auto& materialBuffers = MaterialHandle->GetMaterialBuffers();

			for (uint8_t i = 0; i < materialBuffers.size(); i++)
			{
				if (materialBuffers[i].IsPropertyBuffer) {
					archive(
						cereal::make_nvp("shaderSource", MaterialHandle->GetShader()->GetSourceFilepath()),
						cereal::make_nvp("properties", materialBuffers[i].Value)
					);
					return;
				}
			}

			archive(
				cereal::make_nvp("shaderSource", MaterialHandle->GetShader()->GetSourceFilepath()),
			 	cereal::make_nvp("properties", std::vector<std::byte>())
			);
		}

		template<class Archive>
		void load(Archive& archive)
		{
			std::string shaderSource;
			std::vector<std::byte> buffer;

			archive(
				cereal::make_nvp("shaderSource", shaderSource),
				cereal::make_nvp("properties", buffer)
			);

			MaterialHandle = Ref<Material>::Create(Renderer::shaderLibrary.GetShader(shaderSource));
		    MaterialHandle->SetPropertyBuffer(buffer);
		}
	};
}

#endif // !MATERIAL_COMPONENT_H_