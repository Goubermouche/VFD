#ifndef SPH_SIMULATION_COMPONENT_H_
#define SPH_SIMULATION_COMPONENT_H_

#include "Simulation/SPH/SPHSimulation.h"

namespace fe {
	template<class Archive>
	void serialize(Archive& archive, ParticleVolumeDescription& description)
	{
		archive(
			cereal::make_nvp("sourceMesh", description.sourceMesh),
			cereal::make_nvp("scale", description.scale),
			cereal::make_nvp("offset", description.position),
			cereal::make_nvp("resolution", description.resolution),
			cereal::make_nvp("sampleMode", description.sampleMode)
		);
	}

	template<class Archive>
	void serialize(Archive& archive, SPHSimulationDescription& description)
	{
		archive(
			cereal::make_nvp("timeStep", description.timeStep),
			cereal::make_nvp("globalDamping", description.globalDamping),
			cereal::make_nvp("particleRadius", description.particleRadius),
			cereal::make_nvp("homogenity", description.homogenity),
			cereal::make_nvp("restDensity", description.restDensity),
			cereal::make_nvp("stiffness", description.stiffness),
			cereal::make_nvp("viscosity", description.viscosity),
			cereal::make_nvp("boundsDamping", description.boundsDamping),
			cereal::make_nvp("boundsStiffness", description.boundsStiffness),
			cereal::make_nvp("boundsDampingCritical", description.boundsDampingCritical),
			cereal::make_nvp("maxParticlesInCellCount", description.maxParticlesInCellCount),
			cereal::make_nvp("gravity", description.gravity),
			cereal::make_nvp("worldMin", description.worldMin),
			cereal::make_nvp("worldMax", description.worldMax),
			cereal::make_nvp("particleVolumes", description.particleVolumes)
		);
	}

	struct SPHSimulationComponent
	{
		Ref<SPHSimulation> Handle;

		SPHSimulationComponent() = default;
		SPHSimulationComponent(const SPHSimulationComponent& other) = default;
		SPHSimulationComponent(const SPHSimulationDescription& description)
			: Handle(Ref<SPHSimulation>::Create(description))
		{}
		SPHSimulationComponent(Ref<SPHSimulation> simulation)
			: Handle(simulation)
		{}

		template<class Archive>
		void save(Archive& archive) const
		{
			SPHSimulationDescription description = Handle->GetDescription();
			archive(cereal::make_nvp("description", description));
		}

		template<class Archive>
		void load(Archive& archive)
		{
			SPHSimulationDescription description;
			archive(cereal::make_nvp("description", description));
			Handle = Ref<SPHSimulation>::Create(description);
		}
	}; 
}

#endif // !SPH_SIMULATION_COMPONENT_H_