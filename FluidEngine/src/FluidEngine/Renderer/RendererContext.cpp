#include "pch.h"
#include "RendererContext.h"

#include "RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/OpenGLRendererContext.h"

namespace fe {
	Ref<RendererContext> RendererContext::Create(GLFWwindow* window)
	{
		switch (RendererAPI::GetAPI())
		{
			case RendererAPIType::None:    return nullptr;
			case RendererAPIType::OpenGL:  return Ref<opengl::OpenGLRendererContext>::Create(window);
		}

		ASSERT(false, "unsupported rendering API!");
		return nullptr;
	}
}