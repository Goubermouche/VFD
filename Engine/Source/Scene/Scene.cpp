#include "pch.h"
#include "Scene.h"

#include "Scene/Entity.h"
#include "Scene/Components.h"
#include "Utility/FileSystem.h"


namespace fe {
	void Scene::Save(const std::string& filepath) const 
	{

		try {
			std::ofstream saveFile(filepath.c_str());

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

				SceneData scene;

				output.setNextName("sceneData");
				output(m_Data);
			}

			saveFile.close();
			LOG("scene saved (" + filepath + ")", ConsoleColor::Green);
		}
		catch (const std::exception& exception) {
			ASSERT("error encountered while saving scene!");
			ERR(exception.what(), "scene][save");
		}
	}

	void Scene::Load(const std::string& filepath)
	{
		m_SourceFilepath = filepath;
		m_Name = FilenameFromFilepath(filepath);
		ASSERT(FileExists(m_SourceFilepath), "filepath '" + m_SourceFilepath + "' is invalid!");

		try {
			m_Registry.clear();
			m_EntityIDMap.clear();

			std::ifstream saveFile(m_SourceFilepath.c_str());

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

				input.setNextName("sceneData");
				input(m_Data);
			}

			// Fill the entity ID map
			for (const auto entity : m_Registry.view<IDComponent>()) {
				Entity e = { entity, this };
				m_EntityIDMap[e.GetUUID()] = e;
			}

			saveFile.close();
			LOG("scene loaded (" + filepath + ")", ConsoleColor::Green);
		}
		catch (const std::exception& exception) {
			ASSERT("error encountered while loading scene!");
			ERR(exception.what(), "scene][load");
		}
	}

	Scene::Scene(const std::string& filepath)
		: m_SourceFilepath(filepath)
	{
		ASSERT(FileExists(m_SourceFilepath), "filepath '" + m_SourceFilepath + "' is invalid!");

		try {
			std::ifstream saveFile(m_SourceFilepath.c_str());

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
			for (const auto entity : m_Registry.view<IDComponent>()) {
				Entity e = { entity, this };
				m_EntityIDMap[e.GetUUID()] = e;
			}

			saveFile.close();

			LOG("scene loaded (" + filepath + ")", ConsoleColor::Green);
		}
		catch (const std::exception& exception) {
			ASSERT("error encountered while loading scene!");
			ERR(exception.what(), "scene][load");
		}
	}

	Scene::~Scene()
	{
		m_Registry.clear();
		ERR("clearing registry")
	}

	Entity Scene::CreateEntity(const std::string& name)
	{
		return CreateChildEntity({}, name);
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

	Entity Scene::CreateEntityWithID(const UUID32 id, const std::string& name, bool runtimeMap)
	{
		auto entity = Entity{ m_Registry.create(), this };
		auto& idComponent = entity.AddComponent<IDComponent>();
		idComponent.ID = id;

		entity.AddComponent<TransformComponent>();
		entity.AddComponent<TagComponent>(name.empty() ? "Entity" : name);
		entity.AddComponent<RelationshipComponent>();

		ASSERT(m_EntityIDMap.contains(id), "entity with this id already exists!");
		m_EntityIDMap[id] = entity;
		return entity;
	}

	void Scene::ParentEntity(Entity entity, Entity parent)
	{
		if (parent.IsDescendantOf(entity))
		{
			UnParentEntity(parent);
			Entity newParent = TryGetEntityWithUUID(entity.GetParentUUID());
			if (newParent)
			{
				UnParentEntity(entity);
				ParentEntity(parent, newParent);
			}
		}
		else
		{
			Entity previousParent = TryGetEntityWithUUID(entity.GetParentUUID());
			if (previousParent) {
				UnParentEntity(entity);
			}
		}

		entity.SetParentUUID(parent.GetUUID());
		parent.Children().push_back(entity.GetUUID());
		ConvertToLocalSpace(entity);
	}

	void Scene::UnParentEntity(Entity entity, bool convertToWorldSpace)
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

	void Scene::DeleteEntity(Entity entity,const bool excludeChildren,const bool first)
	{
		if (excludeChildren == false)
		{
			for (size_t i = 0; i < entity.Children().size(); i++)
			{
				auto childId = entity.Children()[i];
				const Entity child = GetEntityWithUUID(childId);
				DeleteEntity(child, excludeChildren, false);
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
		const Entity parent = TryGetEntityWithUUID(entity.GetParentUUID());

		if (!parent) {
			return;
		}

		auto& transform = entity.Transform();
		const glm::mat4 parentTransform = GetWorldSpaceTransformMatrix(parent);
		const glm::mat4 localTransform = glm::inverse(parentTransform) * transform.GetTransform();
		DecomposeTransform(localTransform, transform.Translation, transform.Rotation, transform.Scale);
	}

	void Scene::ConvertToWorldSpace(Entity entity)
	{
		const Entity parent = TryGetEntityWithUUID(entity.GetParentUUID());

		if (!parent) {
			return;
		}

		const glm::mat4 transform = GetWorldSpaceTransformMatrix(entity);
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
		// SPH
		for (const auto entity : m_Registry.view<SPHSimulationComponent, MaterialComponent>()) {
			Entity e = { entity, this };
			auto& material = e.GetComponent<MaterialComponent>();
			auto& simulation = e.GetComponent<SPHSimulationComponent>();
			const float scale = e.Transform().Scale.x;

			const auto& transform = GetWorldSpaceTransformMatrix(e);
			const auto& simulationData = simulation.Handle->GetParameters();

			glm::vec3 worldScale = (simulationData.WorldMaxReal - simulationData.WorldMinReal);
			const glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), { worldScale.x, worldScale.y, worldScale.z });

			material.Handle->Set("model", transform);
			material.Handle->Set("radius", 0.004f * 27.0f * scale);

			// Render domain
			Renderer::DrawBox(transform * scaleMatrix, { 1.0f, 1.0f, 1.0f, 1.0f });
			
			// Render particles
			if (simulationData.ParticleCount > 0) {
				Renderer::DrawPoints(simulation.Handle->GetVAO(), simulationData.ParticleCount, material.Handle);
			}
		}

		// FLIP
		for (const auto entity : m_Registry.view<FLIPSimulationComponent, MaterialComponent>()) {
			Entity e = { entity, this };
			auto& material = e.GetComponent<MaterialComponent>();
			auto& simulation = e.GetComponent<FLIPSimulationComponent>();
			const float scale = e.Transform().Scale.x;

			const auto& transform = GetWorldSpaceTransformMatrix(e);
			const auto& simulationData = simulation.Handle->GetParameters();

			const glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), (glm::vec3)simulationData.Size / 4.0f);

			//material.Handle->Set("model", transform * scaleMatrix);
			//material.Handle->Set("radius", simulationData.ParticleRadius * scaleMatrix * 14.0f);

			//// Render domain
			//// TODO: fix scaling with non-uniformly sized domains 
			//Renderer::DrawBox(glm::translate(transform * scaleMatrix, (glm::vec3)simulationData.Size / 128.0f), {1.0f, 1.0f, 1.0f,1.0f});

			//// Render particles
			//if (simulationData.ParticleCount > 0) {
			//	Renderer::DrawPoints(simulation.Handle->GetVAO(), simulationData.ParticleCount, material.Handle);
			//}

			simulation.Handle->OnRenderTemp();

			
		}


		// Render meshes
		for (const auto entity : m_Registry.view<MeshComponent, MaterialComponent, IDComponent>()) {
			Entity e = { entity, this };
			auto& mesh = e.GetComponent<MeshComponent>();
			auto& material = e.GetComponent<MaterialComponent>();
			// auto& id = e.GetComponent<IDComponent>();

			material.Handle->Set("model", GetWorldSpaceTransformMatrix(e));

			Renderer::DrawTriangles(mesh.Mesh->GetVAO(), mesh.Mesh->GetVertexCount(), material.Handle);
		}
	}

	void Scene::OnUpdate()
	{
		// Update simulations
		// SPH
		for (const entt::entity entity : m_Registry.view<SPHSimulationComponent>()) {
			Entity e = { entity, this };
			auto& simulation = e.GetComponent<SPHSimulationComponent>();

			simulation.Handle->OnUpdate();
		}

		// FLIP
		for (const entt::entity entity : m_Registry.view<FLIPSimulationComponent>()) {
			Entity e = { entity, this };
			auto& simulation = e.GetComponent<FLIPSimulationComponent>();

			simulation.Handle->OnUpdate();
		}
	}

	Entity Scene::GetEntityWithUUID(const UUID32 id) const
	{
		ASSERT(m_EntityIDMap.contains(id), "invalid entity id or entity doesn't exist in the current scene!");
		return m_EntityIDMap.at(id);
	}

	Entity Scene::TryGetEntityWithUUID(const UUID32 id) const
	{
		if (const auto it = m_EntityIDMap.find(id); it != m_EntityIDMap.end()) {
			return it->second;
		}
		return Entity{};
	}

	const std::string& Scene::GetSourceFilepath()
	{
		return m_SourceFilepath;
	}

	const std::string& Scene::GetName()
	{
		return m_Name;
	}

	SceneData& Scene::GetData()
	{
		return m_Data;
	}
}