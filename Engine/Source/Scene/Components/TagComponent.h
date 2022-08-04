#ifndef TAG_COMPONENT_H
#define TAG_COMPONENT_H

#include "pch.h"

namespace fe {
	struct TagComponent {
		std::string Tag;

		TagComponent() = default;
		TagComponent(const TagComponent&) = default;
		TagComponent(const std::string& tag)
			: Tag(tag)
		{}

		operator std::string& () {
			return Tag;
		}

		operator const std::string& () const {
			return Tag;
		}

		template<typename Archive>
		void serialize(Archive& archive) {
			archive(cereal::make_nvp("tag", Tag));
		}
	};
}

#endif // !TAG_COMPONENT_H
