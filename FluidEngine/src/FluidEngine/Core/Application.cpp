#include "pch.h"
#include "Application.h"

// Renderer
#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Renderer/Renderer.h"

// Compute
#include "FluidEngine/Compute/ComputeAPI.h"
#include "FluidEngine/Compute/Compute.h"

#include "FluidEngine/Core/Time.h"
#include "FluidEngine/Platform/CUDA/System.cuh" // temp

namespace fe {
	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		s_Instance = this;

		// Check the current config
#ifdef NDEBUG
		LOG("running in RELEASE")
#else
		LOG("running in DEBUG")
#endif

		// Compute
		{
			Compute::SetAPI(ComputeAPIType::CUDA);
			Compute::Init();
		}
		
		// Renderer
		{
			Renderer::SetAPI(RendererAPIType::OpenGL);

			// Create a new context
			WindowDesc windowDesc;
			windowDesc.width = 1000;
			windowDesc.height = 700;
			windowDesc.title = "window";

			m_Window = std::unique_ptr<Window>(Window::Create(windowDesc));
			m_Window->SetVSync(true);
			m_Window->SetEventCallback([this](Event& e) {
				OnEvent(e);
			});

			Renderer::Init();
		}
		
		
		// Scene
		// m_SceneContext = Scene::Load("res/Scenes/StressTest.json");
		m_SceneContext = Ref<Scene>::Create();
		// Editor
		m_Editor.Reset(new Editor());
		m_Editor->SetSceneContext(m_SceneContext); 

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
		while (m_Running)
		{
			Time::OnUpdate();
			ProcessEvents();
			m_Editor->OnUpdate();
			m_Window->SwapBuffers();
		}
	}

	void Application::Close()
	{
		m_Running = false;
	}

	void Application::SaveScene(const std::string& filePath)
	{
		m_SceneContext->Save(filePath);
	}

	void Application::LoadScene(const std::string& filePath)
	{
		m_SceneContext = Scene::Load(filePath);
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