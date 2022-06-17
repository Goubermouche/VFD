#include "pch.h"
#include "Editor.h"

#include "FluidEngine/Platform/ImGui/ImGuiRenderer.h" 
#include "FluidEngine/Platform/ImGui/ImGuiGLFWBackend.h"
#include "FluidEngine/Core/Application.h"

namespace fe {
	Editor* Editor::s_Instance = nullptr;

	Editor::Editor()
	{
		s_Instance = this;

		InitImGui();

		// Init panels
		m_PanelManager.reset(new PanelManager());
		m_PanelManager->AddPanel<SceneHierarchyPanel>("Scene");
		m_PanelManager->AddPanel<ViewportPanel>("Viewport");
	}

	Editor::~Editor()
	{

	}

	void Editor::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<KeyPressedEvent>([this](KeyPressedEvent& e) {
			return OnKeyPressed(e);
		});

		// Pass unhandled events down to panels
		if (event.Handled == false) {
			m_PanelManager->OnEvent(event);
		}
	}

	void Editor::OnSceneContextChanged(Ref<Scene> context)
	{
		m_PanelManager->OnSceneContextChanged(context);
	}

	void Editor::OnSelectionContextChanged(Entity selectionContext)
	{
		m_SelectionContext = selectionContext;
		m_PanelManager->OnSelectionContextChanged(m_SelectionContext);
	}

	bool Editor::OnKeyPressed(KeyPressedEvent& e)
	{
		switch (e.GetKeyCode())
		{
			// Close application once the escape key is pressed
		case FE_KEY_ESCAPE:
		{
			Application::Get().Close();
			return true; // Stop the event from bubbling further
		}
		}
		return false;
	}

	void Editor::OnUpdate()
	{
		// Begin ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGuiIO& io = ImGui::GetIO();
		Application& app = Application::Get();
		io.DisplaySize = ImVec2((float)Application::Get().GetWindow().GetWidth(), (float)Application::Get().GetWindow().GetHeight());

		// Create an ImGui dockspace
		{
			const ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(viewport->WorkPos);
			ImGui::SetNextWindowSize(viewport->WorkSize);
			ImGui::SetNextWindowViewport(viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f); // TODO: Use scoped style instead
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f); // TODO: Use scoped style instead
			ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDocking;
			windowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			windowFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f)); // TODO: Use scoped style instead
			static bool pOpen = true;

			ImGui::Begin("dockspace", &pOpen, windowFlags);
			ImGui::PopStyleVar(3); // TODO: Use scoped style instead
			ImGuiID dockspaceID = ImGui::GetID("dockspace"); // TODO: Use the UI ID generator
			ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), 0);

			ImGui::End();
			ImGui::PopStyleVar(0);
		}

		// Update panels
		m_PanelManager->OnUpdate();

		// End ImGui frame
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	void Editor::InitImGui()
	{
		// Initialize the ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		// TODO: Create a separate UI context so that support for other platforms can be added - not important right now
		ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow()), true);
		ImGui_ImplOpenGL3_Init("#version 410"); // Use GLSL version 410

		ImGui::StyleColorsDark();

		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

		ImFontConfig config;
		config.OversampleH =5;
		config.OversampleV = 5;
		io.FontDefault = io.Fonts->AddFontFromFileTTF("res/Fonts/Roboto/Roboto-SemiMedium.ttf", 14.0f, &config, io.Fonts->GetGlyphRangesCyrillic());

		io.ConfigWindowsMoveFromTitleBarOnly = true;

	    ImGuiStyle& style = ImGui::GetStyle();
		style.ItemSpacing = { 0, 0 };
		style.WindowPadding = { 0, 0 };
		style.ScrollbarRounding = 2;
		style.FrameBorderSize = 1;
		style.TabRounding = 0;
		style.WindowMenuButtonPosition = ImGuiDir_None;
		style.WindowRounding = 2;

		LOG("ImGui initialized successfully");
	}
}