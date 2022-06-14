#include "pch.h"
#include "VertexArray.h"

#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/OpenGLVertexArray.h"
namespace fe {
	Ref<VertexArray> VertexArray::Create()
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPIType::None:    return nullptr;
		case RendererAPIType::OpenGL:  return Ref<opengl::OpenGLVertexArray>::Create();
		}

		ASSERT(false, "unknown renderer API!");
		return nullptr;
	}
}
