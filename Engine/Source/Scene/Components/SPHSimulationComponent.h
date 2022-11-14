#ifndef SPH_SIMULATION_COMPONENT_H
#define SPH_SIMULATION_COMPONENT_H

#include "pch.h"
#include "Simulation/SPH/SPHSimulation.h"

namespace fe {
	template<class Archive>
	void serialize(Archive& archive, ParticleVolumeDescription& description)
	{
		archive(
			cereal::make_nvp("sourceMesh", description.SourceMesh),
			cereal::make_nvp("scale", description.Scale),
			cereal::make_nvp("offset", description.Position),
			cereal::make_nvp("resolution", description.Resolution),
			cereal::make_nvp("sampleMode", description.SampleMode)
		);
	}

	template<class Archive>
	void serialize(Archive& archive, SPHSimulationDescription& description)
	{
		archive(
			cereal::make_nvp("timeStep", description.TimeStep),
			cereal::make_nvp("globalDamping", description.GlobalDamping),
			cereal::make_nvp("particleRadius", description.ParticleRadius),
			cereal::make_nvp("homogeneity", description.Homogeneity),
			cereal::make_nvp("restDensity", description.RestDensity),
			cereal::make_nvp("stiffness", description.Stiffness),
			cereal::make_nvp("viscosity", description.Viscosity),
			cereal::make_nvp("boundsDamping", description.BoundsDamping),
			cereal::make_nvp("boundsStiffness", description.BoundsStiffness),
			cereal::make_nvp("boundsDampingCritical", description.BoundsDampingCritical),
			cereal::make_nvp("maxParticlesInCellCount", description.MaxParticlesInCellCount),
			cereal::make_nvp("gravity", description.Gravity),
			cereal::make_nvp("worldMin", description.WorldMin),
			cereal::make_nvp("worldMax", description.WorldMax),
			cereal::make_nvp("particleVolumes", description.ParticleVolumes),
			cereal::make_nvp("stepCount", description.StepCount)
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

#endif // !SPH_SIMULATION_COMPONENT_H