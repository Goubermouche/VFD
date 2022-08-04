#ifndef RELATIONSHIP_COMPONENT_H_
#define RELATIONSHIP_COMPONENT_H_

namespace fe {
	struct RelationshipComponent
	{
		UUID32 ParentHandle = 0;
		std::vector<UUID32> Children;

		RelationshipComponent() = default;
		RelationshipComponent(const RelationshipComponent& other) = default;
		RelationshipComponent(const UUID32 parent)
			: ParentHandle(parent)
		{}

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(
				cereal::make_nvp("parent", ParentHandle),
				cereal::make_nvp("children", Children)
			);
		}
	};
}

#endif // !RELATIONSHIP_COMPONENT_H_