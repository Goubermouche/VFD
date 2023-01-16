#include "pch.h"
#include "FluidObjectComponent.h"

namespace vfd
{
	FluidObjectComponent::FluidObjectComponent()
		: Description(FluidObjectDescription{
			false,
			{ 20u, 20u, 20u },
			SampleMode::MaxDensity
		})
	{}

	FluidObjectComponent::FluidObjectComponent(const FluidObjectDescription& description)
		: Description(description)
	{}
}