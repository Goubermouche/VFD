#ifndef GPU_DFSPH_SIMULATION_COMPONENT_H
#define GPU_DFSPH_SIMULATION_COMPONENT_H

#include "Simulation/DFSPH/DFSPHSimulator.h"

namespace vfd {
	template<class Archive>
	void serialize(Archive& archive, DFSPHSimulationDescription& description)
	{
		archive(
			cereal::make_nvp("timeStepSize", description.TimeStepSize),
			cereal::make_nvp("minTimeStepSize", description.MinTimeStepSize),
			cereal::make_nvp("maxTimeStepSize", description.MaxTimeStepSize),
			cereal::make_nvp("frameLength", description.FrameLength),
			cereal::make_nvp("frameCount", description.FrameCount),
			cereal::make_nvp("minPressureSolverIterations", description.MinPressureSolverIterations),
			cereal::make_nvp("maxPressureSolverIterations", description.MaxPressureSolverIterations),
			cereal::make_nvp("maxPressureSolverError", description.MaxPressureSolverError),
			cereal::make_nvp("enableDivergenceSolverError", description.EnableDivergenceSolverError),
			cereal::make_nvp("minDivergenceSolverIterations", description.MinDivergenceSolverIterations),
			cereal::make_nvp("maxDivergenceSolverIterations", description.MaxDivergenceSolverIterations),
			cereal::make_nvp("maxDivergenceSolverError", description.MaxDivergenceSolverError),
			cereal::make_nvp("enableViscositySolver", description.EnableViscositySolver),
			cereal::make_nvp("minViscositySolverIterations", description.MinViscositySolverIterations),
			cereal::make_nvp("maxViscositySolverIterations", description.MaxViscositySolverIterations),
			cereal::make_nvp("maxViscositySolverError", description.MaxViscositySolverError),
			cereal::make_nvp("viscosity", description.Viscosity),
			cereal::make_nvp("boundaryViscosity", description.BoundaryViscosity),
			cereal::make_nvp("tangentialDistanceFactor", description.TangentialDistanceFactor),
			cereal::make_nvp("enableSurfaceTensionSolver", description.EnableSurfaceTensionSolver),
			cereal::make_nvp("surfaceTensionSmoothPassCount", description.SurfaceTensionSmoothPassCount),
			cereal::make_nvp("surfaceTension", description.SurfaceTension),
			cereal::make_nvp("temporalSmoothing", description.TemporalSmoothing),
			cereal::make_nvp("CSDFix", description.CSDFix),
			cereal::make_nvp("CSD", description.CSD),
			cereal::make_nvp("particleRadius", description.ParticleRadius),
			cereal::make_nvp("gravity", description.Gravity)
		);
	}

	struct DFSPHSimulationComponent
	{
		Ref<DFSPHSimulation> Handle;

		DFSPHSimulationComponent();
		DFSPHSimulationComponent(const DFSPHSimulationComponent& other) = default;
		DFSPHSimulationComponent(DFSPHSimulationDescription& description);
		DFSPHSimulationComponent(Ref<DFSPHSimulation> simulation);

		template<class Archive>
		void save(Archive& archive) const;

		template<class Archive>
		void load(Archive& archive);
	};

	template<class Archive>
	inline void DFSPHSimulationComponent::save(Archive& archive) const
	{
		DFSPHSimulationDescription description = Handle->GetDescription();
		archive(cereal::make_nvp("description", description));
	}

	template<class Archive>
	inline void DFSPHSimulationComponent::load(Archive& archive)
	{
		DFSPHSimulationDescription description;
		archive(cereal::make_nvp("description", description));
		Handle = Ref<DFSPHSimulation>::Create(description);
	}
}

#endif // !GPU_DFSPH_SIMULATION_COMPONENT_H