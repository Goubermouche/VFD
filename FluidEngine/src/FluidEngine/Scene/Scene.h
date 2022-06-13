#ifndef SCENE_H_
#define SCENE_H_

#include "entt.hpp"

#include "FluidEngine/Scene/Components.h"

namespace fe {
	class Entity;
	using EntityMap = std::unordered_map<UUID32, Entity>;

	class Scene : public RefCounted
	{
	public:
		Scene();
		~Scene();

		Entity CreateEntity(const std::string& name = "");
		Entity CreateChildEntity(Entity parent, const std::string& name = "");
		Entity CreateEntityWithID(UUID32 UUID32, const std::string& name = "", bool runtimeMap = false);

		void ParentEntity(Entity entity, Entity parent);
		void UnparentEntity(Entity entity, bool convertToWorldSpace = true);

		void DestroyEntity(Entity entity, bool excludeChildren = false, bool first = true);

		void ConvertToLocalSpace(Entity entity);
		void ConvertToWorldSpace(Entity entity);

		glm::mat4 GetWorldSpaceTransformMatrix(Entity entity);

		void OnUpdate();

		Entity GetEntityWithUUID(UUID32 id) const;
		Entity TryGetEntityWithUUID(UUID32 id) const;
		UUID32 GetUUID() const { return m_SceneID; }
	private:
		UUID32 m_SceneID;
		entt::entity m_SceneEntity = entt::null;
		entt::registry m_Registry;

		EntityMap m_EntityIDMap;

		friend class Entity;
	};
}

#endif // !SCENE_H_


