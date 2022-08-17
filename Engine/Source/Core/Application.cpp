#include "pch.h"
#include "Application.h"

// Renderer
#include "Renderer/Renderer.h"

// Compute
#include "Compute/GPUCompute.h"
#include "Core/Time.h"

namespace fe {
	Application* Application::s_Instance = nullptr;
	
	Application::Application()
	{
		s_Instance = this;

		GPUCompute::Init();  

		// Create a new context
		WindowDesc windowDesc;
		windowDesc.Width = 1000;
		windowDesc.Height = 700;
		windowDesc.Title = "Engine";
		windowDesc.VSync = true;

		m_Window = Ref<Window>::Create(windowDesc);
		m_Window->SetEventCallback([this](Event& e) {
			OnEvent(e);
		});

		Renderer::Init();
		
		// Scene
		m_SceneContext = Ref<Scene>::Create(/*"Resources/Scenes/ModelCollection.json"*/);

		// Editor
		m_Editor = Ref<Editor>::Create();
		m_Editor->SetSceneContext(m_SceneContext); 

		//auto simulationEntity = m_SceneContext->CreateEntity("simulation");
		//simulationEntity.Transform().Scale = { 10, 10, 10 };

		//auto& material = simulationEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/PointDiffuseShader.glsl")));
		//material.Handle->Set("color", { 0.73f, 0.73f, 0.73f, 1.0f });

		//SPHSimulationDescription simulationDesc;
		//simulationDesc.ParticleRadius = 0.004f;
		//simulationDesc.Homogeneity = 0.01f;
		//simulationDesc.RestDensity = 1000.0f;
		//simulationDesc.Stiffness = 3.0f;
		//simulationDesc.Viscosity = 1.0f;
		//simulationDesc.MaxParticlesInCellCount = 32;
		//simulationDesc.TimeStep = 0.0016f;
		//simulationDesc.GlobalDamping = 1.0f;
		//simulationDesc.Gravity = { 0.0f, -9.81f, 0.0f };
		//glm::vec3 simulationDomain = { 0.6f, 0.3f, 0.1f };
		//simulationDesc.WorldMin = -simulationDomain;
		//simulationDesc.WorldMax = simulationDomain;
		//simulationDesc.BoundsStiffness = 65536;
		//simulationDesc.BoundsDamping = 256;
		//simulationDesc.BoundsDampingCritical = 60;

		//ParticleVolumeDescription particleDesc1;

		//particleDesc1.SourceMesh = "Resources/Models/Cube.obj";
		//particleDesc1.Scale = { 0.07f, 0.27f, 0.07f };
		//particleDesc1.Position = { 0.5f, 0, 0 };
		//particleDesc1.SampleMode = SampleMode::MaxDensity;
		//particleDesc1.Resolution = { 10, 10, 10 };

		//simulationDesc.ParticleVolumes = { 
		//	particleDesc1
		//};

		//auto& sim = simulationEntity.AddComponent<SPHSimulationComponent>(simulationDesc);
		//sim.Handle->paused = false;
		
		// Mesh test
		Entity meshEntity = m_SceneContext->CreateEntity();
		meshEntity.Transform().Scale = { 1, 1, 1 };
		meshEntity.Transform().Translation = { 0, 0, 0 };
		meshEntity.AddComponent<MeshComponent>("Resources/Models/Cube.obj");
		auto& material = meshEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl")));
		material.Handle->Set("color", {0.4f, 0.4f, 0.4f, 1});

		Run();

		GPUCompute::Shutdown();
		Renderer::ShutDown();
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

		if (event.handled == false) {
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

	void Application::SaveCurrentSceneContext(const std::string& filepath)
	{
		m_SceneContext->Save(filepath);
	}

	void Application::LoadSceneContext(const std::string& filepath)
	{
		m_SceneContext = Ref<Scene>::Create(filepath);
		m_Editor->SetSceneContext(m_SceneContext);
	}

	void Application::ProcessEvents()
	{
		m_Window->ProcessEvents();
		std::scoped_lock<std::mutex> lock(m_EventQueueMutex);

		while (m_EventQueue.empty() == false)
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