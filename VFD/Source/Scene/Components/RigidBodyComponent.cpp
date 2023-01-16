#include "pch.h"
#include "RigidBodyComponent.h"

namespace vfd {
	RigidBodyComponent::RigidBodyComponent()
		 : Description(RigidBodyDescription{
			false,
			0.0f,
			{ 20u, 20u, 20u }
		 })
	{}

	RigidBodyComponent::RigidBodyComponent(const RigidBodyDescription& description)
		 : Description(description)
	{}
}