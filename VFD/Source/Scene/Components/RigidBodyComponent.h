#ifndef STATIC_RIGIDBODY_COMPONENT_H
#define STATIC_RIGIDBODY_COMPONENT_H

#include "Simulation/DFSPH/RigidBody/RigidBody.cuh"

namespace vfd {
	struct RigidBodyComponent
	{
		RigidBodyDescription Description;

		RigidBodyComponent();
		RigidBodyComponent(const RigidBodyComponent& other) = default;
		RigidBodyComponent(const RigidBodyDescription& description);

		template<typename Archive>
		void serialize(Archive& archive);
	};

	template<typename Archive>
	inline void RigidBodyComponent::serialize(Archive& archive)
	{
		archive(
			cereal::make_nvp("inverted", Description.Inverted),
			cereal::make_nvp("padding", Description.Padding),
			cereal::make_nvp("collisionMapResolution", Description.CollisionMapResolution)
		);
	}
}

#endif // !STATIC_RIGIDBODY_COMPONENT_H