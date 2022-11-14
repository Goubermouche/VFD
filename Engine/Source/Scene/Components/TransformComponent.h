#ifndef TRANSFORM_COMPONENT_H
#define TRANSFORM_COMPONENT_H

#include "pch.h"
#include "Core/Math/Math.h"

namespace fe {
	struct TransformComponent {
		glm::vec3 Translation = { 0.0f, 0.0f, 0.0f };
		glm::vec3 Rotation =    { 0.0f, 0.0f, 0.0f };
		glm::vec3 Scale =       { 1.0f, 1.0f, 1.0f };

		TransformComponent() = default;
		TransformComponent(const TransformComponent& other) = default;
		TransformComponent(const glm::vec3& Position)
			: Translation(Position)
		{ }

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

		template<class Archive>
		void save(Archive& archive) const
		{
			archive(cereal::make_nvp("transform", GetTransform()));
		}

		template<class Archive>
		void load(Archive& archive)
		{
			glm::mat4 transform;
			archive(cereal::make_nvp("transform", transform));
			DecomposeTransform(transform, Translation, Rotation, Scale);
		}
	};
}

#endif // !TRANSFORM_COMPONENT_H