#include "pch.h"
#include "FrameBuffer.h"

#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/Buffers/OpenGLFrameBuffer.h"

namespace fe {
	Ref<FrameBuffer> FrameBuffer::Create(const FrameBufferDesc& specification)
	{
		switch (RendererAPI::GetAPIType())
		{
		case RendererAPIType::None:    return nullptr;
		case RendererAPIType::OpenGL:  return Ref<opengl::OpenGLFrameBuffer>::Create(specification);
		}

		ASSERT(false, "unknown renderer API!");
		return nullptr;
	}
}