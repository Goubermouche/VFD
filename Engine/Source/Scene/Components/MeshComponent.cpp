#include "pch.h"
#include "MeshComponent.h"

#include "Core/Application.h"

namespace fe {
	MeshComponent::MeshComponent(const std::string& filepath)
		: Mesh(Ref<TriangleMesh>::Create(filepath))
	{
		Application::Get();
	}
}