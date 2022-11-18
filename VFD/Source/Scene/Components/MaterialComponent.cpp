#include "pch.h"
#include "MaterialComponent.h"

namespace vfd {
	MaterialComponent::MaterialComponent(const Ref<Material> material)
		: Handle(material)
	{}
}