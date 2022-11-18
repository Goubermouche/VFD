#include "pch.h"
#include "StaticRigidBodyComponent.h"

namespace vfd {
	StaticRigidBodyComponent::StaticRigidBodyComponent()
		 : RigidBody(Ref<StaticRigidBody>::Create())
	{}

	StaticRigidBodyComponent::StaticRigidBodyComponent(const StaticRigidBodyDescription& description)
		 : RigidBody(Ref<StaticRigidBody>::Create(description))
	{}
}