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
		m_Window->SetEventCallback([this](Event& e) {
			OnEvent(e);
		});

		glm::vec3 v(1, 4, 5);

		Renderer::Init();

		Run();
	}

	Application::~Application()
	{
	}

	void Application::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<WindowResizeEvent>([this](WindowResizeEvent& e) {
			return OnWindowResize(e); 
		});

		dispatcher.Dispatch<WindowMinimizeEvent>([this](WindowMinimizeEvent& e) { 
			return OnWindowMinimize(e); 
		});

		dispatcher.Dispatch<WindowCloseEvent>([this](WindowCloseEvent& e) { 
			return OnWindowClose(e);
		});

		LOG(event.ToString());
	}

	void Application::Run()
	{
		Renderer::SetClearColor({ 1, 0, 1, 1 });

		while (m_Running)
		{
			ProcessEvents();

			Renderer::Clear();

			m_Window->SwapBuffers();
		}
	}

	void Application::Close()
	{
		m_Running = false;
	}

	void Application::ProcessEvents()
	{
		m_Window->ProcessEvents();

		std::scoped_lock<std::mutex> lock(m_EventQueueMutex);

		// Process custom event queue
		while (m_EventQueue.size() > 0)
		{
			auto& func = m_EventQueue.front();
			func();
			m_EventQueue.pop();
		}
	}

	bool Application::OnWindowResize(WindowResizeEvent& e)
	{
		if (e.GetWidth() == 0 || e.GetHeight() == 0) {
			m_Minimized = true;
			return false;
		}

		m_Minimized = false;
		Renderer::SetViewport(0, 0, e.GetWidth(), e.GetHeight());

		return false;
	}
	bool Application::OnWindowMinimize(WindowMinimizeEvent& e)
	{
		m_Minimized = e.IsMinimized();
		return false;
	}
	bool Application::OnWindowClose(WindowCloseEvent& e)
	{
		Close();
		return false;
	}
}