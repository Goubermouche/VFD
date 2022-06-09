#include "pch.h"
#include "Application.h"
#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Renderer/Renderer.h"

namespace fe {
	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		s_Instance = this;

		RendererAPI::SetAPI(RendererAPIType::OpenGL);

		// Create a new window
		WindowDesc windowDesc;
		windowDesc.width = 1000;
		windowDesc.height = 700;
		windowDesc.title = "window";

		m_Window = std::unique_ptr<Window>(Window::Create(windowDesc));

		Renderer::Init();

		Run();
	}

	Application::~Application()
	{
	}

	void Application::Run()
	{
		Renderer::SetClearColor({ 1, 0, 1, 1 });

		while (m_Running)
		{
			Renderer::Clear();
			m_Window->OnUpdate();
		}
	}
}