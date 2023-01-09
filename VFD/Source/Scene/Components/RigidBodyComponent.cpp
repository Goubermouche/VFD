#include "pch.h"
#include "RigidBodyComponent.h"

namespace vfd {
	RigidBodyComponent::RigidBodyComponent()
		 : Description(RigidBodyDescription{
			false,
			0.0f,
			{20, 20, 20}
		 })
	{}

	RigidBodyComponent::RigidBodyComponent(const RigidBodyDescription& description)
		 : Description(description)
	{}
}