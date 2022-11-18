#ifndef STATIC_RIGIDBODY_COMPONENT_H
#define STATIC_RIGIDBODY_COMPONENT_H

#include "Simulation/DFSPH/StaticBoundarySimulator.h"

namespace vfd {
	struct StaticRigidBodyComponent
	{
		Ref<StaticRigidBody> RigidBody;

		StaticRigidBodyComponent();
		StaticRigidBodyComponent(const StaticRigidBodyComponent& other) = default;
		StaticRigidBodyComponent(const StaticRigidBodyDescription& description);

		// TODO: add saving 
	};
}

#endif // !STATIC_RIGIDBODY_COMPONENT_H