#include "pch.h"
#include "SPHSimulationComponent.h"

namespace vfd {
	SPHSimulationComponent::SPHSimulationComponent()
		: Handle(Ref<SPHSimulation>::Create(
			SPHSimulationDescription{
				0.004f,
				0.01f,
				1000.0f,
				3.0f,
				1.0f,
				0.0016f,
				1.0f,
				65536.0f,
				256.0f,
				60.0f,
				32u,
				2u,
				{ 0.0f, -9.81f, 0.0f },
				{ -0.3f, -0.5f, -0.3f },
				{  0.3f,  0.5f,  0.3f },
				std::vector<ParticleVolumeDescription>
				{
					ParticleVolumeDescription {
						{ 0.2f, 0.2f, 0.2f },
						{ 0.0f, 0.0f, 0.0f },
						{ 20, 20, 20 },
						SampleMode::MaxDensity,
						"Resources/Models/Sphere.obj"
					}
				}
			}
		))
	{}

	SPHSimulationComponent::SPHSimulationComponent(const SPHSimulationDescription& description)
		: Handle(Ref<SPHSimulation>::Create(description))
	{}

	SPHSimulationComponent::SPHSimulationComponent(Ref<SPHSimulation> simulation)
		: Handle(simulation)
	{}
}