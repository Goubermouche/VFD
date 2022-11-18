#include "pch.h"
#include "Application.h"

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Core/Time.h"

namespace fe {
	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		s_Instance = this;


		// Create a new context
		WindowDescription windowDesc;
		windowDesc.Width = 1000;
		windowDesc.Height = 700;
		windowDesc.Title = "vfd";
		windowDesc.VSync = true;

		m_Window = Ref<Window>::Create(windowDesc);
		m_Window->SetEventCallback([this](Event& e) {
			OnEvent(e);
		});

		std::cout <<
			"                ad88           88     \n"
			"               d8\"             88    \n"
			"               88              88     \n"
			"8b       d8  MM88MMM   ,adPPYb,88     \n"
			"`8b     d8'    88     a8\"    `Y88    \n"
			" `8b   d8'     88     8b       88     \n"
			"  `8b,d8'      88     \"8a,   ,d88    \n"
			"    \"8\"        88      `\"8bbdP\"Y8 \n\n"
			"**Viscous Fluid Dynamics information  \n\n"
			"**Enviroment information              \n";

		GPUCompute::Init();
		Renderer::Init();

		m_AssetManager = Ref<AssetManager>::Create();
		m_SceneContext = Ref<Scene>::Create();

		// m_SceneContext->Load("C:/dev/FluidEngine/FluidEngine/Engine/Resources/Scenes/ModelCollection.json");
		//{
		//	auto simulationEntity = m_SceneContext->CreateEntity("simulation");
		//	simulationEntity.Transform().Scale = { 10, 10, 10 };
		//	simulationEntity.Transform().Translation = { 0, 10, 0 };

		//	auto& material = simulationEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/PointDiffuseShader.glsl")));
		//	material.Handle->Set("color", { 0.73f, 0.73f, 0.73f, 1.0f });

		//	SPHSimulationDescription simulationDesc;
		//	simulationDesc.ParticleRadius = 0.004f;
		//	simulationDesc.Homogeneity = 0.01f;
		//	simulationDesc.RestDensity = 1000.0f;
		//	simulationDesc.Stiffness = 3.0f;
		//	simulationDesc.Viscosity = 1.0f;
		//	simulationDesc.MaxParticlesInCellCount = 32;
		//	simulationDesc.TimeStep = 0.0016f;
		//	simulationDesc.GlobalDamping = 1.0f;
		//	simulationDesc.Gravity = { 0.0f, -9.81f, 0.0f };
		//	glm::vec3 simulationDomain = { 0.5, 0.5, 1.0 };
		//	simulationDesc.WorldMin = -simulationDomain;
		//	simulationDesc.WorldMax = simulationDomain;
		//	simulationDesc.BoundsStiffness = 65536;
		//	simulationDesc.BoundsDamping = 256;
		//	simulationDesc.BoundsDampingCritical = 60;
		//	simulationDesc.StepCount = 1;

		//	ParticleVolumeDescription particleDesc1;

		//	particleDesc1.SourceMesh = "Resources/Models/Cube.obj";
		//	particleDesc1.Scale = { 0.475f, 0.475f, 0.1f };
		//	particleDesc1.Position = { 0, 0, -0.595f };
		//	particleDesc1.SampleMode = SampleMode::MaxDensity;
		//	particleDesc1.Resolution = { 10, 10, 10 };

		//	simulationDesc.ParticleVolumes = {
		//		particleDesc1
		//	};

		//	auto& sim = simulationEntity.AddComponent<SPHSimulationComponent>(simulationDesc);
		//	sim.Handle->paused = true;
		//}

		// Mesh test
		//{
		//	Entity meshEntity = m_SceneContext->CreateEntity("Diffuse");
		//	meshEntity.Transform().Scale = { 1, 1, 1 };
		//	meshEntity.Transform().Translation = { 0, 0, 0 };
		//	meshEntity.AddComponent<MeshComponent>("Resources/Models/Sphere.obj");
		//	auto& material = meshEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl")));
		//	material.Handle->Set("color", { 0.4f, 0.4f, 0.4f, 1 });
		//}

		{
			auto simulationEntity = m_SceneContext->CreateEntity("simulation");
			simulationEntity.Transform().Scale = { 10, 10, 10 };
			simulationEntity.Transform().Translation = { 0, 10, 0 };

			auto& material = simulationEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/PointDiffuseShader.glsl")));
			material.Handle->Set("color", { 0.73f, 0.73f, 0.73f, 1.0f });

			DFSPHSimulationDescription simulationDesc;
			simulationDesc.ParticleRadius = 0.025f;
			simulationDesc.CFLMinTimeStepSize = 0.0001f;
			simulationDesc.CFLMaxTimeStepSize = 0.005f;
			simulationDesc.Gravity = { 0.0f, -9.81f, 0.0f };
			simulationDesc.MinPressureSolverIteratations = 2;
			simulationDesc.MaxPressureSolverIterations = 100;
			simulationDesc.MaxPressureSolverError = 0.01f;
			simulationDesc.EnableDivergenceSolver = true;
			simulationDesc.MaxVolumeSolverIterations = 100;
			simulationDesc.MaxVolumeError = 0.1f;

			simulationEntity.AddComponent<DFSPHSimulationComponent>(simulationDesc);
		}
		
		// Editor
		m_Editor = Ref<Editor>::Create();
		m_Editor->SetSceneContext(m_SceneContext);

		Run();

		GPUCompute::Shutdown();
		Renderer::ShutDown();
	}

	void Application::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<WindowResizeEvent>(BIND_EVENT_FN(OnWindowResize));
		dispatcher.Dispatch<WindowMinimizeEvent>(BIND_EVENT_FN(OnWindowMinimize));
		dispatcher.Dispatch<WindowCloseEvent>(BIND_EVENT_FN(OnWindowClose));

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

	Application& Application::Get()
	{
		return *s_Instance;
	}

	Window& Application::GetWindow()
	{
		return *m_Window;
	}

	Ref<Scene>& Application::GetSceneContext()
	{
		return m_SceneContext;
	}

	Ref<AssetManager>& Application::GetAssetManager()
	{
		return m_AssetManager;
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