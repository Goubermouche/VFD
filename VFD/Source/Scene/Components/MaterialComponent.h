#ifndef MATERIAL_COMPONENT_H
#define MATERIAL_COMPONENT_H

#include "Renderer/Renderer.h"

namespace vfd {
	struct MaterialComponent {
		Ref<Material> Handle;

		MaterialComponent() = default;
		MaterialComponent(const MaterialComponent& other) = default;
		MaterialComponent(const Ref<Material> material);
		
		template<class Archive>
		void save(Archive& archive) const;
		
		template<class Archive>
		void load(Archive& archive);
	};

	template<class Archive>
	inline void MaterialComponent::save(Archive& archive) const
	{
		const auto& materialBuffers = Handle->GetMaterialBuffers();

		for (uint8_t i = 0; i < materialBuffers.size(); i++)
		{
			if (materialBuffers[i].IsPropertyBuffer) {
				archive(
					cereal::make_nvp("shaderSource", Handle->GetShader()->GetSourceFilepath()),
					cereal::make_nvp("properties", materialBuffers[i].Value)
				);
				return;
			}
		}

		archive(
			cereal::make_nvp("shaderSource", Handle->GetShader()->GetSourceFilepath()),
			cereal::make_nvp("properties", std::vector<std::byte>())
		);
	}

	template<class Archive>
	inline void MaterialComponent::load(Archive& archive)
	{
		std::string shaderSource;
		std::vector<std::byte> buffer;

		archive(
			cereal::make_nvp("shaderSource", shaderSource),
			cereal::make_nvp("properties", buffer)
		);

		Handle = Ref<Material>::Create(Renderer::GetShader(shaderSource));
		Handle->SetPropertyBuffer(buffer);
	}
}

#endif // !MATERIAL_COMPONENT_H