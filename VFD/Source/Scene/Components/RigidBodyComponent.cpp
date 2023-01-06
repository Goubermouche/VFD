#include "pch.h"
#include "RigidBodyComponent.h"

namespace vfd {
	RigidBodyComponent::RigidBodyComponent()
		 : Handle(RigidBodyDescription{
			glm::mat4(1.0f),
			{20, 20, 20},
			"Resources/Models/Bunny.obj",
			false,
			0.0f
		 })
	{}

	RigidBodyComponent::RigidBodyComponent(const RigidBodyDescription& description)
		 : Handle(description)
	{}
}