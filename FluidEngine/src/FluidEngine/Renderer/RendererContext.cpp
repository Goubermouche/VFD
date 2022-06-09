#include "pch.h"
#include "RendererContext.h"
#include "RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/OpenGLRendererContext.h"

namespace fe {
	std::shared_ptr<RendererContext> RendererContext::Create(GLFWwindow* window)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPIType::None:    return nullptr;
		case RendererAPIType::OpenGL:  return std::shared_ptr<OpenGLRendererContext>(new OpenGLRendererContext(window));
		}

		return nullptr;
	}
}

