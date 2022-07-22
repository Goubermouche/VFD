#ifndef COMPONENTS_H_
#define COMPONENTS_H_

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <types/vector.hpp>	

#include "FluidEngine/Core/Cryptography/UUID.h"
#include "FluidEngine/Core/Math/Math.h"
#include "FluidEngine/Core/Math/GlmSerialization.h"

#include "FluidEngine/Renderer/Renderer.h"

// Components
#include "FluidEngine/Scene/Components/SPHSimulationComponent.h"

#include "FluidEngine/Renderer/Mesh/TriangleMesh.h" // TEMP

namespace fe {
	// This file contains all components. 
	// How to add new components: 
	// 1. Create a new component.
	// 2. Add a serialize() function to it. 
	// 3. Add the component to Save() and Load() functions in Scene.cpp, so they can be saved and loaded properly.

	struct IDComponent
	{
		UUID32 ID = 0;

		template<typename Archive>
		void serialize(Archive& archive) {
			archive(ID);
		}
	};
	
	struct TagComponent {
		std::string Tag;

		TagComponent() = default;
		TagComponent(const TagComponent&) = default;
		TagComponent(const std::string& tag)
			: Tag(tag) {}

		operator std::string& () { return Tag; }
		operator const std::string& () const { return Tag; }

		template<typename Archive>
		void serialize(Archive& archive) {
			archive(Tag);
		}
	};

	struct RelationshipComponent
	{
		UUID32 ParentHandle = 0;
		std::vector<UUID32> Children;

		RelationshipComponent() = default;
		RelationshipComponent(const RelationshipComponent& other) = default;
		RelationshipComponent(UUID32 parent)
			: ParentHandle(parent) {}

		template<typename Archive>
		void serialize(Archive& archive) {
			archive(ParentHandle, Children);
		}
	};

	struct TransformComponent {
		glm::vec3 Translation = { 0.0f, 0.0f, 0.0f };
		glm::vec3 Rotation = { 0.0f, 0.0f, 0.0f };
		glm::vec3 Scale = { 1.0f, 1.0f, 1.0f };

		TransformComponent() = default;
		TransformComponent(const TransformComponent& other) = default;
		TransformComponent(const glm::vec3& translation)
			: Translation(translation) {}

		glm::mat4 GetTransform() const
		{
			return glm::translate(glm::mat4(1.0f), Translation)
				* glm::toMat4(glm::quat(Rotation))
				* glm::scale(glm::mat4(1.0f), Scale);
		}

		void SetTransform(const glm::mat4& transform)
		{
			DecomposeTransform(transform, Translation, Rotation, Scale);
		}

		template<typename Archive>
		void serialize(Archive& archive) {
			archive(GetTransform());
		}
	};

	struct MaterialComponent {
		Ref<Material> Mat;

		MaterialComponent() = default;
		MaterialComponent(const MaterialComponent& other) = default;
		MaterialComponent(Ref<Material> material)
			: Mat(material) {}
		MaterialComponent(const std::string& filePath) // shader filepath
			: Mat(Material::Create(Shader::Create(filePath))) {}
	};

	// TODO: serialization
	// TODO: mehs factory
	struct MeshComponent {
		Ref<TriangleMesh> Mesh;

		MeshComponent() = default;
		MeshComponent(const MeshComponent& other) = default;
		MeshComponent(const std::string& filePath)
			: Mesh(Ref<TriangleMesh>::Create(filePath))
		{}
	};
}

#endif // !COMPONENTS_H_