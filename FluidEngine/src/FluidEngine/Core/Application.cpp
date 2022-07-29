#include "pch.h"
#include "Application.h"

// Renderer
#include "FluidEngine/Renderer/Renderer.h"

// Compute
#include "FluidEngine/Compute/GPUCompute.h"
#include "FluidEngine/Core/Time.h"

namespace fe {
	Application* Application::s_Instance = nullptr;
	
	Application::Application()
	{
		s_Instance = this;

		GPUCompute::Init();  

		// Create a new context
		WindowDesc windowDesc;
		windowDesc.width = 1000;
		windowDesc.height = 700;
		windowDesc.title = "Fluid Engine";
		windowDesc.VSync = false;

		m_Window = Ref<Window>::Create(windowDesc);
		m_Window->SetEventCallback([this](Event& e) {
			OnEvent(e);
		});

		Renderer::Init();
		
		// Scene
		m_SceneContext = Ref<Scene>::Create();

		// Editor
		m_Editor = Ref<Editor>::Create();
		m_Editor->SetSceneContext(m_SceneContext); 

		Run();

		GPUCompute::Shutdown();
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
		while (m_Running)
		{
			Time::OnUpdate();
			ProcessEvents();
			m_SceneContext->OnUpdate();
			m_Editor->OnUpdate();
			m_Window->SwapBuffers();
		}
	}

	void Application::Close()
	{
		m_Running = false;
	}

	void Application::SaveSceneContext(const std::string& filepath)
	{
		m_SceneContext->Save(filepath);
	}

	void Application::LoadScene(const std::string& filepath)
	{
		m_SceneContext = Ref<Scene>::Create(filepath);
		m_Editor->SetSceneContext(m_SceneContext);
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