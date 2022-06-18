#ifndef SCENE_H_
#define SCENE_H_

#include "entt.hpp"

#include "FluidEngine/Scene/Components.h"

namespace fe {
	class Entity;
	using EntityMap = std::unordered_map<UUID32, Entity>;

	/// <summary>
	/// Entity registry wrapper.
	/// </summary>
	class Scene : public RefCounted
	{
	public:
		Scene();
		~Scene();

		Entity CreateEntity(const std::string& name = "");
		Entity CreateChildEntity(Entity parent, const std::string& name = "");
		Entity CreateEntityWithID(UUID32 UUID32, const std::string& name = "", bool runtimeMap = false);

		void Save(const std::string& filePath);
		static Ref<Scene> Load(const std::string& filePath);

		void ParentEntity(Entity entity, Entity parent);
		void UnparentEntity(Entity entity, bool convertToWorldSpace = true);

		void DestroyEntity(Entity entity, bool excludeChildren = false, bool first = true);

		void ConvertToLocalSpace(Entity entity);
		void ConvertToWorldSpace(Entity entity);

		glm::mat4 GetWorldSpaceTransformMatrix(Entity entity);

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
		uint32_t GetEntityCount() {
			// We have to subtract 1, since we don't want to include the scene entity
			return m_Registry.size() - 1;
		}
	private:
		UUID32 m_SceneID;
		entt::registry m_Registry;

		EntityMap m_EntityIDMap;

		friend class Entity;
		friend class SceneHierarchyPanel;
	};
}

#endif // !SCENE_H_


