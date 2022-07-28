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
			const Buffer* buffer = nullptr;

			for (size_t i = 0; i < materialBuffers.size(); i++)
			{
				if (materialBuffers[i].IsPropertyBuffer) {
					buffer = &materialBuffers[i].StorageBuffer;
					break;
				}
			}

			std::vector<byte> bytes;
			if (buffer != nullptr) {
				bytes = std::vector<byte>(static_cast<byte*>(buffer->Data), static_cast<byte*>(buffer->Data) + buffer->Size);
			}

			archive(
				cereal::make_nvp("shaderSource", MaterialHandle->GetShader()->GetSourceFilepath()),
			 	cereal::make_nvp("properties", bytes)
			);

			LOG(buffer->Size, ConsoleColor::Cyan);
		}

		template<class Archive>
		void load(Archive& archive)
		{
			std::string shaderSource;
			std::vector<byte> bufferVector;

			archive(
				cereal::make_nvp("shaderSource", shaderSource),
				cereal::make_nvp("properties", bufferVector)
			);

			void* bufferData = bufferVector.data();
			Buffer buffer(bufferData, bufferVector.size());

			MaterialHandle = Ref<Material>::Create(Ref<Shader>::Create(shaderSource));
		    MaterialHandle->SetPropertyBuffer(buffer);
		}
	};
}

#endif // !MATERIAL_COMPONENT_H_