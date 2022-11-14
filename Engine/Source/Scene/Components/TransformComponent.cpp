#include "pch.h"
#include "TransformComponent.h"

namespace fe {
	TransformComponent::TransformComponent(const glm::vec3& Position)
		: Translation(Position)
	{}

	glm::mat4 TransformComponent::GetTransform() const
	{
		return glm::translate(glm::mat4(1.0f), Translation)
			 * glm::toMat4(glm::quat(Rotation))
			 * glm::scale(glm::mat4(1.0f), Scale);
	}

	void TransformComponent::SetTransform(const glm::mat4& transform)
	{
		DecomposeTransform(transform, Translation, Rotation, Scale);
	}
}