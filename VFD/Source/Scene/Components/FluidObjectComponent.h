#ifndef FLUID_OBJECT_COMPONENT_H
#define FLUID_OBJECT_COMPONENT_H

#include "Simulation/DFSPH/FluidObject/FluidObject.h"

namespace vfd
{
	struct FluidObjectComponent
	{
		FluidObjectDescription Description;

		FluidObjectComponent();
		FluidObjectComponent(const FluidObjectComponent& other) = default;
		FluidObjectComponent(const FluidObjectDescription& description);

		template<typename Archive>
		void serialize(Archive& archive);
	};

	template<typename Archive>
	inline void FluidObjectComponent::serialize(Archive& archive)
	{
		archive(
			cereal::make_nvp("inverted", Description.Inverted),
			cereal::make_nvp("resolution", Description.Resolution),
			cereal::make_nvp("sampleMode", Description.SampleMode)
		);
	}
}

#endif // !FLUID_OBJECT_COMPONENT_H