#ifndef SCENE_H
#define SCENE_H

#include <entt.hpp>
#include <archives/json.hpp>

#include "Core/Cryptography/UUID.h"
#include "Scene/Components.h"

namespace vfd {
	// TODO: add support for multiple panels
	struct SceneData {
		glm::vec3 CameraPosition;
		glm::vec3 CameraPivot;
		std::string ReadMe;

		template<typename Archive>
		void serialize(Archive& archive) {
			archive(
				cereal::make_nvp("cameraPosition", CameraPosition), 
				cereal::make_nvp("cameraPivot", CameraPivot),
				cereal::make_nvp("readMe", ReadMe)
			);
		} 
	};

	class Entity;

	/// <summary>
	/// Entity registry wrapper.
	/// </summary>
	class Scene : public RefCounted
	{
	public:
		Scene() = default;
		Scene(const std::string& filepath);
		~Scene();

		Entity CreateEntity(const std::string& name = "");
		Entity CreateChildEntity(Entity parent, const std::string& name = "");
		Entity CreateEntityWithID(UUID32 id, const std::string& name = "");

		void Save(const std::string& filepath) const;
		void Load(const std::string& filepath);
		void ParentEntity(Entity entity, Entity parent);
		void UnParentEntity(Entity entity, bool convertToWorldSpace = true);

		void DeleteEntity(Entity entity, bool excludeChildren = false, bool first = true);

		void ConvertToLocalSpace(Entity entity);
		void ConvertToWorldSpace(Entity entity);

		glm::mat4 GetWorldSpaceTransformMatrix(Entity entity);

		void OnRender();
		void OnUpdate();
 
		Entity GetEntityWithUUID(UUID32 id) const;
		Entity TryGetEntityWithUUID(UUID32 id) const;

		UUID32 GetUUID() const;
		uint32_t GetEntityCount() const;

		template<typename Component, typename... Other, typename... Exclude>
		entt::basic_view<entt::registry::entity_type, entt::get_t<Component, Other...>, entt::exclude_t<Exclude...>> View(entt::exclude_t<Exclude...> = {});

		const std::string& GetSourceFilepath();
		SceneData& GetData();
	private:
		using EntityMap = std::unordered_map<UUID32, Entity>;

		UUID32 m_SceneID;
		entt::registry m_Registry;;
		EntityMap m_EntityIDMap;

		std::string m_SourceFilepath;
		SceneData m_Data;

		friend class Entity;
		friend class SceneHierarchyPanel;
	};

	template<typename Component, typename ...Other, typename ...Exclude>
	inline entt::basic_view<entt::registry::entity_type, entt::get_t<Component, Other...>, entt::exclude_t<Exclude...>> Scene::View(entt::exclude_t<Exclude...>)
	{
		return m_Registry.view<Component, Other..., Exclude...>();
	}
}

#endif // !SCENE_H