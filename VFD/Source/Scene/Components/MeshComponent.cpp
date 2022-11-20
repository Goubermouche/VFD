#include "pch.h"
#include "MeshComponent.h"

#include "Core/Application.h"

namespace vfd {
	MeshComponent::MeshComponent(const std::string& filepath)
	{
	    Mesh = Application::Get().GetAssetManager()->GetOrCreateAsset<MeshAsset>(filepath)->GetMesh();
	}
}