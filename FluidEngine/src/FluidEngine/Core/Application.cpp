#include "pch.h"
#include "Application.h"
#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Renderer/Renderer.h"

namespace fe {
	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		s_Instance = this;

		// Scene
		m_SceneContext = Ref<Scene>::Create();

		// Entity test
		auto e = m_SceneContext->CreateEntity();
		m_SceneContext->CreateChildEntity(e);

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

		Renderer::Init();

		// Editor
		m_Editor.Reset(new Editor());
		
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

		if (event.Handled == false) {
			m_Editor->OnEvent(event);
		}
	}

	void Application::Run()
	{

		Renderer::SetClearColor({ 0, 0, 0, 1 });

		while (m_Running)
		{
			ProcessEvents();
			Renderer::SetClearColor({ std::sin(glfwGetTime()), 0, 0, 1});
			Renderer::Clear();
			m_Editor->Ondate();

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