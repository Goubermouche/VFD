#include "pch.h"
#include "IndexBuffer.h"

#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/Buffers/OpenGLIndexBuffer.h"

namespace fe {
	Ref<IndexBuffer> IndexBuffer::Create(std::vector<uint32_t>& indices)
	{
		switch (RendererAPI::GetAPIType())
		{
		case RendererAPIType::None:    return nullptr;
		case RendererAPIType::OpenGL:  return Ref<opengl::OpenGLIndexBuffer>::Create(indices);
		}

		ASSERT(false, "unsupported rendering API!");
		return nullptr;
	}
}