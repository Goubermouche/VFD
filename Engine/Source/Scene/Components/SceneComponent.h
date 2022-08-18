#ifndef SCENE_COMPONENT_H
#define SCENE_COMPONENT_H

namespace fe {
	struct SceneComponent
	{
		SceneComponent() = default;
		SceneComponent(const SceneComponent& other) = default;

		template<class Archive>
		void save(Archive& archive) const
		{
			//SPHSimulationDescription description = Handle->GetDescription();
			//archive(cereal::make_nvp("description", description));
		}

		template<class Archive>
		void load(Archive& archive)
		{
			//SPHSimulationDescription description;
			//archive(cereal::make_nvp("description", description));
			//Handle = Ref<SPHSimulation>::Create(description);
		}
	};
}

#endif // !SCENE_COMPONENT_H