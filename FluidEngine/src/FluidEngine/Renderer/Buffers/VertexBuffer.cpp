#include "pch.h"
#include "VertexBuffer.h"

#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/Buffers/OpenGLVertexBuffer.h"

namespace fe {
	Ref<VertexBuffer> VertexBuffer::Create(std::vector<float>& vertices)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPIType::None:    return nullptr;
		case RendererAPIType::OpenGL:  return Ref<opengl::OpenGLVertexBuffer>::Create(vertices);
		}

		ASSERT(false, "unsupported rendering API!");
		return nullptr;
	}
}