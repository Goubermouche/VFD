#include "pch.h"
#include "TagComponent.h"

namespace vfd {
	TagComponent::TagComponent(const std::string& tag)
		: Tag(tag)
	{}

	TagComponent::operator std::string& ()
	{
		return Tag;
	}

	TagComponent::operator const std::string& () const
	{
		return Tag;
	}
}