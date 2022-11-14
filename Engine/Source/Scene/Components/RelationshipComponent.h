#ifndef RELATIONSHIP_COMPONENT_H
#define RELATIONSHIP_COMPONENT_H

#include "Core/Cryptography/UUID.h"

namespace fe {
	struct RelationshipComponent
	{
		UUID32 ParentHandle = 0;
		std::vector<UUID32> Children;

		RelationshipComponent() = default;
		RelationshipComponent(const RelationshipComponent& other) = default;
		RelationshipComponent(const UUID32 parent);

		template<class Archive>
		void serialize(Archive& archive);
	};

	template<class Archive>
	inline void RelationshipComponent::serialize(Archive& archive)
	{
		archive(
			cereal::make_nvp("parent", ParentHandle),
			cereal::make_nvp("children", Children)
		);
	}
}

#endif // !RELATIONSHIP_COMPONENT_H