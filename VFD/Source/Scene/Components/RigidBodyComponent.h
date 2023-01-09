#ifndef STATIC_RIGIDBODY_COMPONENT_H
#define STATIC_RIGIDBODY_COMPONENT_H

#include "Simulation/GPUDFSPH/RigidBody/RigidBody.cuh"

namespace vfd {
	struct RigidBodyComponent
	{
		RigidBodyDescription Description;

		RigidBodyComponent();
		RigidBodyComponent(const RigidBodyComponent& other) = default;
		RigidBodyComponent(const RigidBodyDescription& description);

		// TODO: add saving 
	};
}

#endif // !STATIC_RIGIDBODY_COMPONENT_H