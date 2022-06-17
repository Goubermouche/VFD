#include "pch.h"
#include "Scene.h"

#include "FluidEngine/Scene/Entity.h"
#include "FluidEngine/Scene/Components.h"

namespace fe {

	class Archive {
	public: 
		Archive(std::fstream& os)
		: m_FileOutput(os) {
		}

		void operator()(entt::entt_traits<unsigned int>::entity_type ent) {
			m_FileOutput << ent << std::endl;
		}

		void operator()(entt::entity ent) {
			m_FileOutput << entt::to_integral(ent) << std::endl;

		}
		template <typename T>
		void operator()(entt::entity ent, const T& data) {
			m_FileOutput << entt::to_integral(ent) << std::endl;
		}
	private:
		std::fstream& m_FileOutput;
	};

	Scene::Scene()
	{
	}

	Scene::~Scene()
	{
	}

	Entity Scene::CreateEntity(const std::string& name)
	{
		return CreateChildEntity({}, name);;
	}

	Entity Scene::CreateChildEntity(Entity parent, const std::string& name)
	{
		auto entity = Entity{ m_Registry.create(), this };
		auto& idComponent = entity.AddComponent<IDComponent>();
		idComponent.ID = {};

		entity.AddComponent<TransformComponent>();
		entity.AddComponent<TagComponent>(name.empty() ? "Entity" : name);
		entity.AddComponent<RelationshipComponent>();

		if (parent) {
			entity.SetParent(parent);
		}

		m_EntityIDMap[idComponent.ID] = entity;
		return entity;
	}

	Entity Scene::CreateEntityWithID(UUID32 UUID32, const std::string& name, bool runtimeMap)
	{
		auto entity = Entity{ m_Registry.create(), this };
		auto& idComponent = entity.AddComponent<IDComponent>();
		idComponent.ID = UUID32;

		entity.AddComponent<TransformComponent>();
		entity.AddComponent<TagComponent>(name.empty() ? "Entity" : name);
		entity.AddComponent<RelationshipComponent>();

		ASSERT(m_EntityIDMap.find(UUID32) == m_EntityIDMap.end(), "entity with this id already exists!");
		m_EntityIDMap[UUID32] = entity;
		return entity;
	}

	void Scene::ParentEntity(Entity entity, Entity parent)
	{
		if (parent.IsDescendantOf(entity))
		{
			UnparentEntity(parent);

			Entity newParent = TryGetEntityWithUUID(entity.GetParentUUID());
			if (newParent)
			{
				UnparentEntity(entity);
				ParentEntity(parent, newParent);
			}
		}
		else
		{
			Entity previousParent = TryGetEntityWithUUID(entity.GetParentUUID());

			if (previousParent) {
				UnparentEntity(entity);
			}
		}

		entity.SetParentUUID(parent.GetUUID());
		parent.Children().push_back(entity.GetUUID());

		ConvertToLocalSpace(entity);
	}

	void Scene::UnparentEntity(Entity entity, bool convertToWorldSpace)
	{
		Entity parent = TryGetEntityWithUUID(entity.GetParentUUID());
		if (!parent) {
			return;
		}

		auto& parentChildren = parent.Children();
		parentChildren.erase(std::remove(parentChildren.begin(), parentChildren.end(), entity.GetUUID()), parentChildren.end());

		if (convertToWorldSpace) {
			ConvertToWorldSpace(entity);
		}

		entity.SetParentUUID(0);
	}

	void Scene::DestroyEntity(Entity entity, bool excludeChildren, bool first)
	{
		if (!excludeChildren)
		{
			for (size_t i = 0; i < entity.Children().size(); i++)
			{
				auto childId = entity.Children()[i];
				Entity child = GetEntityWithUUID(childId);
				DestroyEntity(child, excludeChildren, false);
			}
		}

		if (first)
		{
			if (auto parent = entity.GetParent(); parent) {
				parent.RemoveChild(entity);
			}
		}

		m_EntityIDMap.erase(entity.GetUUID());
		m_Registry.destroy(entity.m_EntityHandle);
	}

	void Scene::ConvertToLocalSpace(Entity entity)
	{
		Entity parent = TryGetEntityWithUUID(entity.GetParentUUID());

		if (!parent) {
			return;
		}

		auto& transform = entity.Transform();
		glm::mat4 parentTransform = GetWorldSpaceTransformMatrix(parent);

		glm::mat4 localTransform = glm::inverse(parentTransform) * transform.GetTransform();
		DecomposeTransform(localTransform, transform.Translation, transform.Rotation, transform.Scale);
	}

	void Scene::ConvertToWorldSpace(Entity entity)
	{
		Entity parent = TryGetEntityWithUUID(entity.GetParentUUID());

		if (!parent) {
			return;
		}

		glm::mat4 transform = GetWorldSpaceTransformMatrix(entity);
		auto& entityTransform = entity.Transform();
		DecomposeTransform(transform, entityTransform.Translation, entityTransform.Rotation, entityTransform.Scale);
	}

	glm::mat4 Scene::GetWorldSpaceTransformMatrix(Entity entity)
	{
		glm::mat4 transform(1.0f);

		Entity parent = TryGetEntityWithUUID(entity.GetParentUUID());
		if (parent) {
			transform = GetWorldSpaceTransformMatrix(parent);
		}

		return transform * entity.Transform().GetTransform();
	}
	void Scene::OnUpdate()
	{
	}

	void Scene::Save(const std::string& filePath)
	{
		WARN(filePath);

		std::fstream file;
		file.open(filePath);
		Archive archive(file);

		entt::snapshot{ m_Registry }
		.entities(archive)
		.component<
			IDComponent, 
			TagComponent,
			RelationshipComponent, 
			TransformComponent
		>(archive);

		file.close();

		WARN("REGISTRY SAVED");
	}

	void Scene::Load(const std::string& filePath)
	{
		//entt::snapshot_loader{m_Registry}.entities()
	}

	Entity Scene::GetEntityWithUUID(UUID32 id) const
	{
		ASSERT(m_EntityIDMap.find(id) != m_EntityIDMap.end(), "invalid entity id or entity doesn't exist in the current scene!");
		return m_EntityIDMap.at(id); return Entity();
	}

	Entity Scene::TryGetEntityWithUUID(UUID32 id) const
	{
		if (const auto iter = m_EntityIDMap.find(id); iter != m_EntityIDMap.end()) {
			return iter->second;
		}
		return Entity{};
	}
}