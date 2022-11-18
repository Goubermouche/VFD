#ifndef ASSET_REGISTRY_H
#define ASSET_REGISTRY_H

#include "Scene/Asset.h"

#include "Scene/Assets/TextureAsset.h"
#include "Scene/Assets/MeshAsset.h"

namespace fe {
	class AssetManager : public RefCounted
	{
	public:
		template<typename T, typename ...Args>
		Ref<T> CreateAsset(const std::string& filepath, Args&&... args);

		UUID32 GetAssetIDFromFilepath(const std::string& filepath);

		template<typename T>
		Ref<T> GetAsset(const UUID32 id);

		template<typename T>
		Ref<T> GetAsset(const std::string& filepath);

		template<typename T, typename ...Args>
		Ref<T> GetOrCreateAsset(const std::string& filepath, Args&&... args);

	private:

		std::unordered_map<UUID32, Ref<Asset>> m_AssetRegistry;
	};

	template<typename T, typename ...Args>
	inline Ref<T> AssetManager::CreateAsset(const std::string& filepath, Args && ...args)
	{
		static_assert(std::is_base_of<Asset, T>::value, "class does not inherit from Asset!");

		Ref<T> asset = Ref<T>::Create(filepath, args...);
		m_AssetRegistry[UUID32()] = asset;

		// LOG("asset created successfully (" + filepath + ")", "asset manager", ConsoleColor::Green);
		return asset;
	}

	template<typename T>
	inline Ref<T> AssetManager::GetAsset(const UUID32 id)
	{
		ASSERT(m_AssetRegistry.contains(id), "asset registry does not contain the specified handle! (" + std::to_string(id) + ")");
		return Ref<T>(m_AssetRegistry[id]);
	}

	template<typename T>
	inline Ref<T> AssetManager::GetAsset(const std::string& filepath)
	{
		return GetAsset<T>(GetAssetIDFromFilepath(filepath));
	}

	template<typename T, typename ...Args>
	inline Ref<T> AssetManager::GetOrCreateAsset(const std::string& filepath, Args && ...args)
	{
		for (const auto& [key, value] : m_AssetRegistry) {
			if (value->GetSourceFilepath() == filepath) {
				return value;
			}
		}

		return CreateAsset<T>(filepath, args...);
	}
}

#endif // !ASSET_REGISTRY_H