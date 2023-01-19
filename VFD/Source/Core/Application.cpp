#include "pch.h"
#include "Application.h"

#include "Debug/SystemInfo.h"
#include "Renderer/Renderer.h"
#include "Core/Time.h"

namespace vfd {
	Application* Application::s_Instance = nullptr;

	struct Test
	{
		glm::vec3 Property;
	};

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
			"**Viscous Fluid Dynamics information  \n"
			"Version: 0.3.4                        \n\n"
			"**Environment information             \n";

		SystemInfo::Init();
		Renderer::Init();

		m_AssetManager = Ref<AssetManager>::Create();
		m_SceneContext = Ref<Scene>::Create();

		{
			{
				auto simulationEntity = m_SceneContext->CreateEntity("GPU Simulation");

				DFSPHSimulationDescription simulationDesc;

				// Time step
				simulationDesc.TimeStepSize = 0.001f;
				simulationDesc.MinTimeStepSize = 0.0001f;
				simulationDesc.MaxTimeStepSize = 0.005f;

				simulationDesc.FrameLength = 0.016f;
				simulationDesc.FrameCount = 200u;

				// Pressure solver
				simulationDesc.MinPressureSolverIterations = 0u;
				simulationDesc.MaxPressureSolverIterations = 100u;
				simulationDesc.MaxPressureSolverError = 10.0f;

				// Divergence solver
				simulationDesc.EnableDivergenceSolverError = true;
				simulationDesc.MinDivergenceSolverIterations = 0u;
				simulationDesc.MaxDivergenceSolverIterations = 100u;
				simulationDesc.MaxDivergenceSolverError = 10.0f;

				// Viscosity solver
				simulationDesc.MinViscositySolverIterations = 0u;
				simulationDesc.MaxViscositySolverIterations = 100u;
				simulationDesc.MaxViscositySolverError = 0.1f;
				simulationDesc.Viscosity = 10.0f;
				simulationDesc.BoundaryViscosity = 10.0f;
				simulationDesc.TangentialDistanceFactor = 0.5f;

				// Surface tension
				simulationDesc.EnableSurfaceTensionSolver = false;

				// Scene
				simulationDesc.ParticleRadius = 0.025f;
				simulationDesc.Gravity = { 0.0f, -9.81f, 0.0f };
				// simulationDesc.Gravity = { 0.0f, 0.0f, 0.0f };

				auto& material = simulationEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/DFSPHParticleSimpleShader.glsl")));
				material.Handle->Set("color", { 0.0f, 0.2f, 0.976f, 1.0f });

				auto& simulation = simulationEntity.AddComponent<DFSPHSimulationComponent>(simulationDesc);
			}

			{
				auto boundaryEntity = m_SceneContext->CreateEntity("Rigid Body");
				boundaryEntity.Transform().Scale = { 2.0f, 0.2f, 2.0f };

				RigidBodyDescription rigidBodyDesc;

				rigidBodyDesc.CollisionMapResolution = { 20u, 20u, 20u };
				rigidBodyDesc.Inverted = false;
				rigidBodyDesc.Padding = 0.0f;

				boundaryEntity.AddComponent<RigidBodyComponent>(rigidBodyDesc);
				boundaryEntity.AddComponent<MeshComponent>("Resources/Models/Cube.obj");

				auto& material = boundaryEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl")));
				material.Handle->Set("color", { 0.4f, 0.4f, 0.4f, 1.0f });
			}

			{
				auto fluidObjectEntity = m_SceneContext->CreateEntity("Fluid Object");
				fluidObjectEntity.Transform().Translation = { 0.0f, 3.0f, 0.0f };
				fluidObjectEntity.Transform().Rotation = { glm::radians(180.0f), 0.0f, 0.0f };

				FluidObjectDescription fluidObjectDesc;

				fluidObjectDesc.Resolution = { 20u, 20u, 20u };
				fluidObjectDesc.Inverted = false;
				fluidObjectDesc.SampleMode = SampleMode::MediumDensity;

				fluidObjectEntity.AddComponent<FluidObjectComponent>(fluidObjectDesc);
				fluidObjectEntity.AddComponent<MeshComponent>("Resources/Models/Cone.obj");

				auto& material = fluidObjectEntity.AddComponent<MaterialComponent>(Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/ColorShader.glsl")));
				material.Handle->Set("color", { 1.0f, 1.0, 1.0f, 1.0f });
			}
		}

		// Editor
		m_Editor = Ref<Editor>::Create();
		m_Editor->SetSceneContext(m_SceneContext);

		Run();

		SystemInfo::Shutdown();
		Renderer::Shutdown();
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