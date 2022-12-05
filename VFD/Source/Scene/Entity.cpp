#include "pch.h"
#include "Entity.h"

namespace vfd {
	Entity::Entity(const entt::entity handle, Scene* scene)
		: m_EntityHandle(handle), m_Scene(scene)
	{ }

	TransformComponent& Entity::Transform()
	{
		return m_Scene->m_Registry.get<TransformComponent>(m_EntityHandle);
	}

	const glm::mat4& Entity::Transform() const
	{
		return m_Scene->m_Registry.get<TransformComponent>(m_EntityHandle).GetTransform();
	}

	Entity::operator uint32_t() const
	{
		return static_cast<unsigned>(m_EntityHandle);
	}

	Entity::operator entt::entity() const
	{
		return m_EntityHandle;
	}

	Entity::operator bool() const
	{
		return (m_EntityHandle != entt::null) && m_Scene;
	}

	bool Entity::operator==(const Entity& other) const
	{
		return m_EntityHandle == other.m_EntityHandle && m_Scene == other.m_Scene;
	}

	bool Entity::operator!=(const Entity& other) const
	{
		return !(*this == other);
	}

	Entity Entity::GetParent()
	{
		return m_Scene->TryGetEntityWithUUID(GetParentUUID());
	}

	void Entity::SetParent(Entity parent)
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

	void Entity::SetParentUUID(UUID32 parent)
	{
		GetComponent<RelationshipComponent>().ParentHandle = parent;
	}

	UUID32 Entity::GetParentUUID() const
	{
		return GetComponent<RelationshipComponent>().ParentHandle;
	}

	std::vector<UUID32>& Entity::Children()
	{
		return GetComponent<RelationshipComponent>().Children;
	}

	bool Entity::RemoveChild(Entity child)
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

	bool Entity::IsAncestorOf(Entity entity)
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

	bool Entity::IsDescendantOf(Entity entity) const
	{
		return entity.IsAncestorOf(*this);
	}

	UUID32 Entity::GetUUID()
	{
		return GetComponent<IDComponent>().ID;
	}

	UUID32 Entity::GetSceneUUID()
	{
		return m_Scene->GetUUID();
	}
}