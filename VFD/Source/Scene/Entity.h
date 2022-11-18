#ifndef ENTITY_H
#define ENTITY_H

#include "Scene/Scene.h"

#define ENTITY_NAME_MAX_LENGTH 28 // chars

namespace vfd {
	class Entity
	{
	public:
		Entity() = default;
		Entity(const entt::entity handle, Scene* scene);
		Entity(const Entity& other) = default;
		~Entity() = default;

		template<typename T, typename... Args>
		T& AddComponent(Args&&... args);

		template<typename T>
		T& GetComponent();

		template<typename T>
		const T& GetComponent() const;

		template<typename T>
		bool HasComponent();

		template<typename... T>
		bool HasComponent() const;

		template<typename T>
		void RemoveComponent();

		TransformComponent& Transform();
		const glm::mat4& Transform() const;

		operator uint32_t () const;
		operator entt::entity() const;
		operator bool() const;
		bool operator ==(const Entity& other) const;
		bool operator !=(const Entity& other) const;

		Entity GetParent();
		void SetParent(Entity parent);
		void SetParentUUID(UUID32 parent);
		UUID32 GetParentUUID() const;

		std::vector<UUID32>& Children();
		bool RemoveChild(Entity child);

		bool IsAncestorOf(Entity entity);
		bool IsDescendantOf(Entity entity) const;

		UUID32 GetUUID();
		UUID32 GetSceneUUID();
	private:
		entt::entity m_EntityHandle{ entt::null };
		Ref<Scene> m_Scene = nullptr;

		friend class Scene;
	};

	template<typename T, typename ...Args>
	inline T& Entity::AddComponent(Args && ...args)
	{
		ASSERT(!HasComponent<T>(), "entity already contains the " + std::string(typeid(T).name()) + " component!");
		return m_Scene->m_Registry.emplace<T>(m_EntityHandle, std::forward<Args>(args)...);
	}

	template<typename T>
	inline T& Entity::GetComponent()
	{
		ASSERT(HasComponent<T>(), "entity does not contains the " + std::string(typeid(T).name()) + " component!");
		return m_Scene->m_Registry.get<T>(m_EntityHandle);
	}

	template<typename T>
	inline const T& Entity::GetComponent() const
	{
		ASSERT(HasComponent<T>(), "entity does not contains the " + std::string(typeid(T).name()) + " component!");
		return m_Scene->m_Registry.get<T>(m_EntityHandle);
	}

	template<typename T>
	inline bool Entity::HasComponent()
	{
		return m_Scene->m_Registry.any_of<T>(m_EntityHandle);
	}

	template<typename ...T>
	inline bool Entity::HasComponent() const
	{
		return m_Scene->m_Registry.any_of<T...>(m_EntityHandle);
	}

	template<typename T>
	inline void Entity::RemoveComponent()
	{
		ASSERT(HasComponent<T>(), "entity does not contains the " + std::string(typeid(T).name()) + " component!");
		m_Scene->m_Registry.remove<T>(m_EntityHandle);
	}
}

#endif // !ENTITY_H