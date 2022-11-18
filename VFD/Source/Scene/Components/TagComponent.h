#ifndef TAG_COMPONENT_H
#define TAG_COMPONENT_H

namespace vfd {
	struct TagComponent {
		std::string Tag;

		TagComponent() = default;
		TagComponent(const TagComponent&) = default;
		TagComponent(const std::string& tag);
			
		operator std::string& ();
		operator const std::string& () const;

		template<typename Archive>
		void serialize(Archive& archive);
	};

	template<typename Archive>
	inline void TagComponent::serialize(Archive& archive)
	{
		archive(cereal::make_nvp("tag", Tag));
	}
}

#endif // !TAG_COMPONENT_H
