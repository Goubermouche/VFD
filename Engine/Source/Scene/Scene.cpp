#include "pch.h"
#include "Scene.h"

#include "Scene/Entity.h"
#include "Scene/Components.h"
#include "Utility/FileSystem.h"

#include <archives/json.hpp>

namespace fe {
	void Scene::Save(const std::string& filePath)
	{
		try {
			std::ofstream saveFile(filePath.c_str());

			// Since cereal's Input archive finishes saving data in the destructor we have to place it in a separate scope.
			{
				cereal::JSONOutputArchive output{ saveFile };
				entt::snapshot{ m_Registry }
					.entities(output)
					.component<
					IDComponent,
					TagComponent,
					RelationshipComponent,
					TransformComponent,
					SPHSimulationComponent,
					MaterialComponent,
					MeshComponent
					>(output);
			}

			saveFile.close();
			LOG("scene saved to '" + filePath + "'");
		}
		catch (const std::exception& exception) {
			ASSERT("error encountered while saving scene!");
			ERR(exception.what(), "scene][save");
		}
	}

	Scene::Scene()
	{}

	Scene::Scene(const std::string& filepath)
		: m_SourceFilePath(filepath)
	{
		ASSERT(FileExists(m_SourceFilePath), "filepath '" + m_SourceFilePath + "' is invalid!");

		try {
			std::ifstream saveFile(m_SourceFilePath.c_str());

			std::stringstream saveFileData;
			saveFileData << saveFile.rdbuf();

			// Since cereal's Input archive finishes loading data in the destructor we have to place it in a separate scope.
			{
				cereal::JSONInputArchive input{ saveFileData };
				entt::snapshot_loader{ m_Registry }
					.entities(input)
					.component<
					IDComponent,
					TagComponent,
					RelationshipComponent,
					TransformComponent,
					SPHSimulationComponent,
					MaterialComponent,
					MeshComponent
					>(input);
			}

			// Fill the entity ID map
			for (auto entity : m_Registry.view<IDComponent>()) {
				Entity e = { entity, this };
				m_EntityIDMap[e.GetUUID()] = e;
			}

			saveFile.close();

			LOG("scene loaded from '" + m_SourceFilePath + "'");
		}
		catch (const std::exception& exception) {
			ASSERT("error encountered while loading scene!");
			ERR(exception.what(), "scene][load");
		}
	}

	Scene::~Scene()
	{}

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

	void Scene::OnRender()
	{
		// Render simulations
		for (auto entity : m_Registry.view<SPHSimulationComponent, MaterialComponent>()) {
			Entity e = { entity, this };
			auto& material = e.GetComponent<MaterialComponent>();
			auto& simulation = e.GetComponent<SPHSimulationComponent>();
			const float scale = e.Transform().Scale.x;

			const auto& transform = GetWorldSpaceTransformMatrix(e);
			const auto& simulationData = simulation.Handle->GetData();

			glm::vec3 worldScale = (simulationData.worldMaxReal - simulationData.worldMinReal);
			const glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), { worldScale.x, worldScale.y, worldScale.z });

			material.Handle->Set("model", transform);
			material.Handle->Set("radius", 0.004f * 27.0f * scale);

			// Render domain
			Renderer::DrawBox(transform * scaleMatrix, { 1.0f, 1.0f, 1.0f, 1.0f });
			
			// Render particles
			if (simulationData.particleCount > 0) {
				Renderer::DrawPoints(simulation.Handle->GetVAO(), simulationData.particleCount, material.Handle);
			}
		}

		// Render meshes
		for (auto entity : m_Registry.view<MeshComponent, MaterialComponent>()) {
			Entity e = { entity, this };
			auto& mesh = e.GetComponent<MeshComponent>();
			auto& material = e.GetComponent<MaterialComponent>();

			material.Handle->Set("model", GetWorldSpaceTransformMatrix(e));

			Renderer::DrawTriangles(mesh.Mesh->GetVAO(), mesh.Mesh->GetVertexCount(), material.Handle);
		}
	}

	void Scene::OnUpdate()
	{
		// Update simulations
		for (auto entity : m_Registry.view<SPHSimulationComponent>()) {
			Entity e = { entity, this };
			auto& simulation = e.GetComponent<SPHSimulationComponent>();

			simulation.Handle->OnUpdate();
		}
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

	const std::string& Scene::GetSourceFilePath()
	{
		return m_SourceFilePath;
	}
}