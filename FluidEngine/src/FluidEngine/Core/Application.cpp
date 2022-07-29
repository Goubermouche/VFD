#include "pch.h"
#include "Application.h"

// Renderer
#include "FluidEngine/Renderer/Renderer.h"

// Compute
#include "FluidEngine/Compute/GPUCompute.h"
#include "FluidEngine/Core/Time.h"

// temp simulation test
#include "FluidEngine/Simulation/SPH/SPHSimulation.h"

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

		m_Window = std::unique_ptr<Window>(Window::Create(windowDesc));
		m_Window->SetVSync(false);
		m_Window->SetEventCallback([this](Event& e) {
			OnEvent(e);
		});

		Renderer::Init();
		
		// Scene
		// m_SceneContext = Scene::Load("res/Scenes/ModelCollection.json");
		m_SceneContext = Ref<Scene>::Create();

		//auto simulationEntity = m_SceneContext->CreateEntity("simulation");
		//simulationEntity.Transform().Scale = { 10, 10, 10 };

		//auto& material = simulationEntity.AddComponent<MaterialComponent>("res/Shaders/Normal/PointDiffuseShader.glsl");
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
		//glm::vec3 simulationDomain = { 0.3f, 0.3f, 0.3f };
		//simulationDesc.worldMin = -simulationDomain;
		//simulationDesc.worldMax = simulationDomain;
		//simulationDesc.boundsStiffness = 65536;
		//simulationDesc.boundsDamping = 256;
		//simulationDesc.boundsDampingCritical = 60;

		//ParticleVolumeDescription particleDesc1;

		//particleDesc1.sourceMesh = "res/Models/Cube.obj";
		//particleDesc1.scale = { 0.1f, 0.27f, 0.1f };
		//////////particleDesc1.position = { -0.17f, 0.0f, 0.17f };
		//particleDesc1.sampleMode = SampleMode::MaxDensity;
		//particleDesc1.resolution = { 10, 10, 10 };

		//simulationDesc.particleVolumes = { 
		//	particleDesc1,
		//};

		//simulationEntity.AddComponent<SPHSimulationComponent>(simulationDesc);


		
		// Mesh test{
		//{
		//	auto meshEntity = m_SceneContext->CreateEntity("Bunny");
		//	meshEntity.Transform().Scale = { 1, 1, 1 };
		//	meshEntity.AddComponent<MeshComponent>("res/Models/Bunny.obj");
		//	auto& material = meshEntity.AddComponent<MaterialComponent>("res/Shaders/Normal/BasicDiffuseShader.glsl");
		//	material.MaterialHandle->Set("color", { 0.1f, 0.1f, 0.5f, 1.0f });
		//}
	
		//{
		//	auto meshEntity = m_SceneContext->CreateEntity("Sphere");
		//	meshEntity.Transform().Scale = { 1, 1, 1 };
		//	meshEntity.Transform().Translation = { 3, 0, 0 };
		//	meshEntity.AddComponent<MeshComponent>("res/Models/Sphere.obj");
		//	auto& material = meshEntity.AddComponent<MaterialComponent>("res/Shaders/Normal/BasicDiffuseShader.glsl");
		//	material.MaterialHandle->Set("color", { 0.5f, 0.1f, 0.1f, 1.0f });
		//}

		//{
		//	auto meshEntity = m_SceneContext->CreateEntity("Torus");
		//	meshEntity.Transform().Scale = { 1, 1, 1 };
		//	meshEntity.Transform().Translation = { -3, 0, 0 };
		//	meshEntity.AddComponent<MeshComponent>("res/Models/Torus.obj");
		//	auto& material = meshEntity.AddComponent<MaterialComponent>("res/Shaders/Normal/BasicDiffuseShader.glsl");
		//	material.MaterialHandle->Set("color", { 0.1f, 0.5f, 0.1f, 1.0f });
		//}

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