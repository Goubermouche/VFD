#include "pch.h"
#include "MaterialComponent.h"

namespace fe {
	MaterialComponent::MaterialComponent(const Ref<Material> material)
		: Handle(material)
	{}
}