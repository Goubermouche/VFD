#include "pch.h"
#include "Renderer.h"
#include "FluidEngine/Platform/OpenGL/OpenGLRenderer.h"

namespace fe {

	RendererAPI* Renderer::s_RendererAPI = nullptr;

	void Renderer::Init()
	{
		s_RendererAPI = new opengl::OpenGLRenderer;
		s_RendererAPI->Init();
		LOG("renderer initialized successfully");
	}

	void Renderer::SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
	{
		s_RendererAPI->SetViewport(x, y, width, height);
	}

	void Renderer::SetClearColor(const glm::vec4& color)
	{
		s_RendererAPI->SetClearColor(color);
	}

	void Renderer::Clear()
	{
		s_RendererAPI->Clear();
	}
}