#include "pch.h"
#include "Material.h"

#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/OpenGLMaterial.h"

namespace fe {
	Ref<Material> Material::Create(const Ref<Shader>& shader, const std::string& name)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPIType::None:    return nullptr;
		case RendererAPIType::OpenGL:  return Ref<opengl::OpenGLMaterial>::Create(shader, name);
		}

		ASSERT(false, "unknown renderer API!");
		return nullptr;
	}
}