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
		windowDesc.Title = "Fluid Engine";
		windowDesc.VSync = false;

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

		//auto& material = simulationEntity.AddComponent<MaterialComponent>("Resources/Shaders/Normal/PointDiffuseShader.glsl");
		//material.MaterialHandle->Set("color", { 0.73f, 0.73f, 0.73f, 1.0f });

		//SPHSimulationDescription simulationDesc;
		//simulationDesc.particleRadius = 0.004f;
		//simulationDesc.homogenity = 0.01f;
		//simulationDesc.restDensity = 1000.0f;
		//simulationDesc.stiffness = 3.0f;
		//simulationDesc.viscosity = 0.5f;
		//simulationDesc.maxParticlesInCellCount = 32;
		//simulationDesc.timeStep = 0.0016f;
		//simulationDesc.globalDamping = 1.0f;
		//simulationDesc.gravity = { 0.0f, -9.81f, 0.0f };
		//glm::vec3 simulationDomain = { 0.3f, 1.0f, 0.3f };
		//simulationDesc.worldMin = -simulationDomain;
		//simulationDesc.worldMax = simulationDomain;
		//simulationDesc.boundsStiffness = 65536;
		//simulationDesc.boundsDamping = 256;
		//simulationDesc.boundsDampingCritical = 60;

		//ParticleVolumeDescription particleDesc1;
		//ParticleVolumeDescription particleDesc2;

		//particleDesc1.sourceMesh = "Resources/Models/Cube.obj";
		//particleDesc1.scale = { 0.273f, 0.02f, 0.273f };
		//particleDesc1.position = { 0, -0.95f, 0 };
		//particleDesc1.sampleMode = SampleMode::MaxDensity;
		//particleDesc1.resolution = { 10, 10, 10 };

		//particleDesc2.sourceMesh = "Resources/Models/Sphere.obj";
		//particleDesc2.scale = { 0.1f, 0.1f, 0.1f };
		//particleDesc2.position = { 0, 0.8f, 0 };
		//particleDesc2.sampleMode = SampleMode::MaxDensity;
		//particleDesc2.resolution = { 10, 10, 10 };

		//simulationDesc.particleVolumes = { 
		//	particleDesc1,
		//	particleDesc2
		//};


		//simulationEntity.AddComponent<SPHSimulationComponent>(simulationDesc);

			// Mesh test
		//auto meshEntity = m_SceneContext->CreateEntity("bunny");
		//meshEntity.Transform().Scale = { 3, 3, 3 };
		//meshEntity.AddComponent<MeshComponent>("Resources/Models/ObjectCollection.obj");
		//auto& material = meshEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::shaderLibrary.GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl")));
		//material.MaterialHandle->Set("color", {0.4f, 0.4f, 0.4f, 1});


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