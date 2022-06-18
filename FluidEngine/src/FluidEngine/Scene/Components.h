#ifndef COMPONENTS_H_
#define COMPONENTS_H_

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include "FluidEngine/Core/Cryptography/UUID.h"
#include "FluidEngine/Core/Math/Math.h"

#include "FluidEngine/Core/Math/Math.h"
#include "FluidEngine/Core/Math/GlmConversions.h"

#include <types/vector.hpp>	

namespace fe {
	struct IDComponent
	{
		UUID32 ID = 0;
	};
	
	struct TagComponent {
		std::string Tag;

		TagComponent() = default;
		TagComponent(const TagComponent&) = default;
		TagComponent(const std::string& tag)
			: Tag(tag) {}

		operator std::string& () { return Tag; }
		operator const std::string& () const { return Tag; }
	};

	struct RelationshipComponent
	{
		UUID32 ParentHandle = 0;
		std::vector<UUID32> Children;

		RelationshipComponent() = default;
		RelationshipComponent(const RelationshipComponent& other) = default;
		RelationshipComponent(UUID32 parent)
			: ParentHandle(parent) {}
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
	};

	template<typename Archive>
	void serialize(Archive& archive, IDComponent& idComponent) {
		archive(idComponent.ID);
	}

	template<typename Archive>
	void serialize(Archive& archive, TagComponent& tagComponent) {
		archive(tagComponent.Tag);
	}

	template<typename Archive>
	void serialize(Archive& archive, RelationshipComponent& relationShipComponent) {
		//std::vector<uint32_t> childIds(relationShipComponent.Children.begin(), relationShipComponent.Children.end());
		archive(relationShipComponent.ParentHandle, relationShipComponent.Children);
	}

	template<typename Archive>
	void serialize(Archive& archive, TransformComponent& transformComponent) {
		archive(transformComponent.GetTransform());
	}
}

#endif // !COMPONENTS_H_
