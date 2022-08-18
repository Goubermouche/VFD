#ifndef SCENE_H
#define SCENE_H

#include "entt.hpp"

#include "Scene/Components.h"

namespace fe {
	class Entity;
	using EntityMap = std::unordered_map<UUID32, Entity>;

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
		Entity CreateEntityWithID(UUID32 id, const std::string& name = "", bool runtimeMap = false);

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

		/// <summary>
		/// Gets the scene ID.
		/// </summary>
		/// <returns>Scene ID.</returns>
		UUID32 GetUUID() const { 
			return m_SceneID; 
		}

		/// <summary>
		/// Gets the count of entities situated in the scene, excluding the scene entity.
		/// </summary>
		/// <returns>Number of entities in the scene.</returns>
		uint32_t GetEntityCount() const{
			return m_Registry.alive();
		}

		template<typename Component, typename... Other, typename... Exclude>
		entt::basic_view<entt::registry::entity_type, entt::get_t<Component, Other...>, entt::exclude_t<Exclude...>> View(entt::exclude_t<Exclude...> = {}) {
			return m_Registry.view<Component, Other..., Exclude...>();
		}

		const std::string& GetSourceFilePath();
	private:
		UUID32 m_SceneID;
		entt::registry m_Registry;;
		EntityMap m_EntityIDMap;
		std::string m_SourceFilePath;

		friend class Entity;
		friend class SceneHierarchyPanel;
	};
}

#endif // !SCENE_H