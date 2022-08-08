#ifndef ENTITY_H
#define ENTITY_H

#include "pch.h"
#include "Scene/Components.h"
#include "Scene/Scene.h"

#define ENTITY_NAME_MAX_LENGTH 32 // chars

namespace fe {
	class Entity
	{
	public:
		Entity() = default;
		Entity(const entt::entity handle, Scene* scene)
			: m_EntityHandle(handle), m_Scene(scene)
		{}
		Entity(const Entity& other) = default;
		~Entity() = default;

		template<typename T, typename... Args>
		T& AddComponent(Args&&... args) {
			ASSERT(!HasComponent<T>(), "entity already contains the specified component!");
			return m_Scene->m_Registry.emplace<T>(m_EntityHandle, std::forward<Args>(args)...);
		}

		template<typename T>
		T& GetComponent() {
			ASSERT(HasComponent<T>(), "entity does not contain the specified component!");
			return m_Scene->m_Registry.get<T>(m_EntityHandle);
		}

		template<typename T>
		const T& GetComponent() const
		{
			ASSERT(HasComponent<T>(), "entity does not contain the specified component!");
			return m_Scene->m_Registry.get<T>(m_EntityHandle);
		}

		template<typename T>
		bool HasComponent() {
			return m_Scene->m_Registry.any_of<T>(m_EntityHandle);
		}

		template<typename... T>
		bool HasComponent() const
		{
			return m_Scene->m_Registry.any_of<T...>(m_EntityHandle);
		}

		template<typename T>
		void RemoveComponent() {
			ASSERT(HasComponent<T>(), "entity does not have the specified component!");
			m_Scene->m_Registry.remove<T>(m_EntityHandle);
		}

		TransformComponent& Transform() { 
			return m_Scene->m_Registry.get<TransformComponent>(m_EntityHandle); 
		}

		const glm::mat4& Transform() const { 
			return m_Scene->m_Registry.get<TransformComponent>(m_EntityHandle).GetTransform(); 
		}

		operator uint32_t () const { 
			return (uint32_t)m_EntityHandle;
		}

		operator entt::entity() const { 
			return m_EntityHandle;
		}

		operator bool() const {
			return (m_EntityHandle != entt::null) && m_Scene;
		}

		bool operator ==(const Entity& other) const {
			return m_EntityHandle == other.m_EntityHandle && m_Scene == other.m_Scene;
		}

		bool operator !=(const Entity& other) const {
			return !(*this == other);
		}

		Entity GetParent() {
			return m_Scene->TryGetEntityWithUUID(GetParentUUID());
		}

		void SetParent(Entity parent)
		{
			Entity currentParent = GetParent();
			if (currentParent == parent) {
				return;
			}

			if (currentParent) {
				currentParent.RemoveChild(*this);
			}

			SetParentUUID(parent.GetUUID());

			if (parent)
			{
				auto& parentChildren = parent.Children();
				const UUID32 id = GetUUID();
				if (std::ranges::find(parentChildren.begin(), parentChildren.end(), id) == parentChildren.end()) {
					parentChildren.emplace_back(GetUUID());
				}
			}
		}

		void SetParentUUID(UUID32 parent) {
			GetComponent<RelationshipComponent>().ParentHandle = parent; 
		}

		UUID32 GetParentUUID() const { 
			return GetComponent<RelationshipComponent>().ParentHandle;
		}

		std::vector<UUID32>& Children() { 
			return GetComponent<RelationshipComponent>().Children;
		}

		bool RemoveChild(Entity child)
		{
			const UUID32 childId = child.GetUUID();
			std::vector<UUID32>& children = Children();
			const auto it = std::ranges::find(children.begin(), children.end(), childId);
			if (it != children.end())
			{
				children.erase(it);
				return true;
			}

			return false;
		}

		bool IsAncestorOf(Entity entity)
		{
			const auto& children = Children();

			if (children.empty()) {
				return false;
			}

			for (UUID32 child : children)
			{
				if (child == entity.GetUUID()) {
					return true;
				}
			}

			for (const UUID32 child : children)
			{
				if (m_Scene->GetEntityWithUUID(child).IsAncestorOf(entity)) {
					return true;
				}
			}

			return false;
		}

		bool IsDescendantOf(Entity entity) const
		{
			return entity.IsAncestorOf(*this);
		}

		UUID32 GetUUID() { 
			return GetComponent<IDComponent>().ID;
		}

		UUID32 GetSceneUUID() { 
			return m_Scene->GetUUID(); 
		}
	private:
		entt::entity m_EntityHandle{ entt::null };
		Ref<Scene> m_Scene = nullptr;

		friend class Scene;
	};
}

#endif // !ENTITY_H