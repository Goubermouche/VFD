#include "pch.h"
#include "Editor.h"

// Panels
#include "FluidEngine/Editor/Panels/SceneHierarchyPanel.h"
#include "FluidEngine/Editor/Panels/ViewportPanel.h"
#include "FluidEngine/Editor/Panels/SystemInfoPanel.h"

#include "FluidEngine/Platform/ImGui/ImGuiRenderer.h" 
#include "FluidEngine/Platform/ImGui/ImGuiGLFWBackend.h"
#include "FluidEngine/Core/Application.h"
#include "FluidEngine/Utility/FileSystem.h"

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
		m_PanelManager->AddPanel<SystemInfoPanel>("Info");
	}

	Editor::~Editor()
	{
	}

	void Editor::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<KeyPressedEvent>([this](KeyPressedEvent& e) {
			return OnKeyPressed(e);;
		});

		// Pass unhandled events down to panels
		if (event.Handled == false) {
			m_PanelManager->OnEvent(event);
		}
	}

	void Editor::SetSceneContext(Ref<Scene> context)
	{
		m_SceneContext = context;
		m_PanelManager->SetSceneContext(context);
	}

	void Editor::SetSelectionContext(Entity context)
	{
		m_SelectionContext = context;
		m_PanelManager->SetSelectionContext(m_SelectionContext);
	}

	void Editor::SaveScene()
	{
		// Open a save file dialog.
		std::string filePath = FileDialog::SaveFile("Json files (*.json)|*.json|Text files (*.txt)|*.txt", "json");
		if (filePath.empty() == false) {
			// Call the application save API.
			Application::Get().SaveScene(filePath);
		}
	}

	void Editor::LoadScene()
	{
		// Open an open file dialog.
		std::string filePath = FileDialog::OpenFile("Json files (*.json)|*.json|Text files (*.txt)|*.txt");
		if (filePath.empty() == false) {
			// Call the application save API.
			Application::Get().LoadScene(filePath);
		}
	}

	bool Editor::GetCameraMode()
	{
		return m_CameraTrackpadMode;
	}

	// Global editor shortcuts should be placed here.
	// TODO: Potentially replace these if statements by a shortcut data structure.
	bool Editor::OnKeyPressed(KeyPressedEvent& e)
	{
		// Call shortcuts only when the key get pressed.
		if (e.GetRepeatCount() > 0) {
			return false;
		}

		if (Input::IsKeyPressed(FE_KEY_ESCAPE)) {
			Application::Get().Close();
			return true; // Stop the event from bubbling further.
		}

		if (Input::IsKeyPressed(FE_KEY_DELETE)) {
			if (m_SelectionContext) {
				m_SceneContext->DestroyEntity(m_SelectionContext);
				SetSelectionContext({});
			}
			return true;
		}

		// Save the current scene on 'Ctrl + S'
		if (Input::IsKeyPressed(FE_KEY_LEFT_CONTROL)) {
			if (Input::IsKeyPressed(FE_KEY_S)) {
				std::string filePath = m_SceneContext->GetSourceFilePath();

				if (filePath.empty()) {
					SaveScene();
				}
				else {
					Application::Get().SaveScene(filePath);
				}
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
			ImGui::PopStyleVar(3); // TODO: Use scoped style instead ? 
			ImGuiID dockspaceID = ImGui::GetID("dockspace"); // TODO: Use the UI ID generator
			ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), 0);
			ImGui::End();
		}

		// Draw main menu bar
		{
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 0.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 2.0f, 2.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 4.0f });

			if (ImGui::BeginMainMenuBar()) {
				if (ImGui::BeginMenu("File")) {
					if (ImGui::MenuItem("Save")) {
						// Save the current scene
						SaveScene();
					}

					if (ImGui::MenuItem("Open...")) {
						// Load a new scene
						LoadScene();
					}
					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Utility")) {
					if (ImGui::MenuItem("Toggle Style Editor")) {
						m_StyleEditorEnabled = !m_StyleEditorEnabled;
					}

					if (ImGui::MenuItem("Toggle ImGui Demo Window")) {
						m_ImGuiDemoWindowEnabled = !m_ImGuiDemoWindowEnabled;
					}

					ImGui::Separator();

					if (ImGui::MenuItem("Toggle Camera Mode")) {
						m_CameraTrackpadMode = !m_CameraTrackpadMode;
					}
					ImGui::EndMenu();
				}

				ImGui::EndMainMenuBar();
			}

			ImGui::PopStyleVar(3);
		}
		
		// Update panels
		m_PanelManager->OnUpdate();

		// Temp utility functions 
		if (m_StyleEditorEnabled) {
			ImGui::ShowStyleEditor();
		}

		if (m_ImGuiDemoWindowEnabled) {
			ImGui::ShowDemoWindow();
		}

		// End ImGui frame
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	void Editor::InitImGui()
	{
		// Initialize the ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		// TODO: Create a separate UI context so that support for other platforms can be added (?) - not important right now
		ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow()), true);
		ImGui_ImplOpenGL3_Init("#version 410"); // Use GLSL version 410


		// IO
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigWindowsMoveFromTitleBarOnly = true;

		// Font
		ImFontConfig config;
		config.OversampleH =5;
		config.OversampleV = 5;
		io.FontDefault = io.Fonts->AddFontFromFileTTF("res/Fonts/Roboto/Roboto-SemiMedium.ttf", 14.0f, &config, io.Fonts->GetGlyphRangesCyrillic());

		// Style
		ImGui::StyleColorsDark();
	    ImGuiStyle& style = ImGui::GetStyle();
		style.ItemSpacing = { 0.0f, 0.0f };
		style.WindowPadding = { 0.0f, 0.0f };
		style.ScrollbarRounding = 2.0f;
		style.FrameBorderSize = 1.0f;
		style.TabRounding = 0.0f;
		style.WindowMenuButtonPosition = ImGuiDir_None;
		style.WindowRounding = 2.0f;

		LOG("ImGui initialized successfully");
	}
}