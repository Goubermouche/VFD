#include "pch.h"
#include "Application.h"

// Renderer
#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Renderer/Renderer.h"

// Compute
#include "FluidEngine/Compute/GPUCompute.h"
#include "FluidEngine/Core/Time.h"

// temp simulation test
#include "FluidEngine/Simulation/SPH/SPHSimulation.h"

namespace fe {
	Application* Application::s_Instance = nullptr;
	UUID32 simulationEntityHandle;

	Application::Application()
	{
		s_Instance = this;

		GPUCompute::Init();  
		Renderer::SetAPI(RendererAPIType::OpenGL);

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
		m_SceneContext = Ref<Scene>::Create();

		//auto simulationEntity = m_SceneContext->CreateEntity("simulation");

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
		//ParticleVolumeDescription particleDesc2;

		//particleDesc1.sourceMesh = "res/Models/Cube.obj";
		//particleDesc1.scale = { 0.273f, 0.02f, 0.273f };
		//particleDesc1.position = { 0, -0.25f, 0 };
		//particleDesc1.sampleMode = SampleMode::MaxDensity;
		//particleDesc1.resolution = { 10, 10, 10 };

		//particleDesc2.sourceMesh = "res/Models/Sphere.obj";
		//particleDesc2.scale = { 0.1f, 0.1f, 0.1f };
		//particleDesc2.position = { 0, 0, 0 };
		//particleDesc2.sampleMode = SampleMode::MaxDensity;
		//particleDesc2.resolution = { 10, 10, 10 };

		//simulationDesc.particleVolumes = { 
		//	particleDesc1,
		//	particleDesc2
		//};

		// simulationEntity.AddComponent<SPHSimulationComponent>(simulationDesc);
		//simulationEntityHandle = simulationEntity.GetUUID();


		// Mesh test
		//auto meshEntity = m_SceneContext->CreateEntity("bunny");
		//meshEntity.Transform().Scale = { 3, 3, 3 };
		//meshEntity.AddComponent<MeshComponent>("res/Models/ObjectCollection.obj");
		//auto& material = meshEntity.AddComponent<MaterialComponent>("res/Shaders/Normal/BasicDiffuseShader.glsl");
		//material.MaterialHandle->Set("color", {0.4f, 0.4f, 0.4f, 1});
		
		// Editor
		m_Editor.Reset(new Editor());
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