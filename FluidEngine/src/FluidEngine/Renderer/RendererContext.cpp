#include "pch.h"
#include "RendererContext.h"

#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/OpenGLRendererContext.h"

namespace fe {
	Ref<RendererContext> RendererContext::Create(GLFWwindow* window)
	{
		switch (RendererAPI::GetAPIType())
		{
		case RendererAPIType::None:   return nullptr;
		case RendererAPIType::OpenGL: return Ref<opengl::OpenGLRendererContext>::Create(window);
		}

		ASSERT(false, "unsupported rendering API!");
		return nullptr;
	}
}