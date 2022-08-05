#ifndef ASSET_REGISTRY_H
#define ASSET_REGISTRY_H

#include "Scene/Asset.h"
#include "Scene/Assets/TextureAsset.h"

namespace fe {
	class AssetManager : public RefCounted
	{
	public:
		template<typename T, typename ...Args>
		Ref<T> Add(const std::string& filepath, Args&&... args) {
			static_assert(std::is_base_of<Asset, T>::value, "class does not inherit from Asset!");

			Ref<T> asset = Ref<T>::Create(filepath, args...);
			m_AssetRegistry[UUID32()] = asset;

			LOG("asset created successfully (" + filepath + ")", "asset manager", ConsoleColor::Green);
			return asset;
		}

		UUID32 GetAssetUUIDFromFilepath(const std::string& filepath) {
			for (const auto& [key, value] : m_AssetRegistry) {
				if (value->GetSourceFilepath() == filepath) {
					return key;
				}
			}

			ASSERT("asset registry does not contain an asset with the specified file path! (" + filepath + ")");
			return 0;
		}

		template<typename T>
		Ref<T> Get(const UUID32 id) {
			ASSERT(m_AssetRegistry.contains(id), "asset registry does not contain the specified handle! (" + std::to_string(id) + ")");
			return Ref<T>(m_AssetRegistry[id]);
		}

		template<typename T>
		Ref<T> Get(const std::string& filepath) {
			return Get<T>(GetAssetUUIDFromFilepath(filepath));
		}
		
	private:
		std::unordered_map<UUID32, Ref<Asset>> m_AssetRegistry;
	};
}

#endif // !ASSET_REGISTRY_H