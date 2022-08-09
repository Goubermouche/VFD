#include "pch.h"
#include "Editor.h"

// Panels
#include "Editor/Panels/SceneHierarchyPanel.h"
#include "Editor/Panels/ViewportPanel.h"
#include "Editor/Panels/SystemInfoPanel.h"

#include "UI/ImGui/ImGuiRenderer.h" 
#include "UI/ImGui/ImGuiGLFWBackend.h"
#include "Core/Application.h"
#include "Utility/FileSystem.h"

#include "UI/UI.h"

namespace fe {
	Editor* Editor::s_Instance = nullptr;
	
	Editor::Editor()
	{
		s_Instance = this;

		// Init the asset manager
		m_AssetManager = Ref<AssetManager>::Create();
		m_AssetManager->Add<TextureAsset>("Resources/Images/Editor/search.png");
		m_AssetManager->Add<TextureAsset>("Resources/Images/Editor/close.png");
		m_AssetManager->Add<TextureAsset>("Resources/Images/Editor/test.png");
		
		// Init the UI
		UI::Init();

		// Init panels
		m_PanelManager.reset(new PanelManager());
		m_PanelManager->AddPanel<SceneHierarchyPanel>("Scene");
		m_PanelManager->AddPanel<ViewportPanel>("Viewport");
	 	m_PanelManager->AddPanel<SystemInfoPanel>("Info");
	}

	void Editor::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<KeyPressedEvent>([this](KeyPressedEvent& e) {
			return OnKeyPressed(e);;
		});

		// Pass unhandled events down to panels
		if (event.handled == false) {
			m_PanelManager->OnEvent(event);
		}
	}

	void Editor::SetSceneContext(const Ref<Scene> context)
	{
		m_SceneContext = context;
		m_PanelManager->SetSceneContext(context);
	}

	void Editor::SetSelectionContext(const Entity context)
	{
		m_SelectionContext = context;
		m_PanelManager->SetSelectionContext(m_SelectionContext);
	}

	void Editor::SaveCurrentSceneContext()
	{
		// Open a save file dialog.
		const std::string filepath = FileDialog::SaveFile("Json files (*.json)|*.json|Text files (*.txt)|*.txt", "json");
		if (filepath.empty() == false) {
			// Call the application save API.
			Application::Get().SaveCurrentSceneContext(filepath);
		}
	}

	void Editor::LoadSceneContext()
	{
		// Open an open file dialog.
		const std::string filepath = FileDialog::OpenFile("Json files (*.json)|*.json|Text files (*.txt)|*.txt");
		if (filepath.empty() == false) {
			// Call the application save API.
			Application::Get().LoadSceneContext(filepath);
		}
	}

	CameraControlMode Editor::GetCameraMode() const
	{
		return m_CameraControlMode;
	}

	// Global editor shortcuts should be placed here.
	// TODO: Potentially replace these if statements by a shortcut data structure.
	bool Editor::OnKeyPressed(KeyPressedEvent& e)
	{
		// Call shortcuts only when the key get pressed.
		if (e.GetRepeatCount() > 0) {
			return false;
		}

		// Exit the application on 'Escape'
		//if (Input::IsKeyPressed(KEY_ESCAPE)) {
		//	Application::Get().Close();
		//	return true; // Stop the event from bubbling further.
		//}

		// Delete the selected entity on 'Delete'
		if (Input::IsKeyPressed(KEY_DELETE)) {
			if (m_SelectionContext) {
				m_SceneContext->DeleteEntity(m_SelectionContext);
				SetSelectionContext({});
			}
			return true;
		}

		// Save the current scene on 'Ctrl + S'
		if (Input::IsKeyPressed(KEY_LEFT_CONTROL)) {
			if (Input::IsKeyPressed(KEY_S)) {
				const std::string filepath = m_SceneContext->GetSourceFilePath();

				if (filepath.empty()) {
					SaveCurrentSceneContext();
				}
				else {
					Application::Get().SaveCurrentSceneContext(filepath);
				}
			}
		}

		if (Input::IsKeyPressed(KEY_SPACE)) {
			for (const entt::entity entity : m_SceneContext->View<SPHSimulationComponent>()) {
				Entity e = { entity, m_SceneContext.Raw()};
				auto& simulation = e.GetComponent<SPHSimulationComponent>();

				simulation.Handle->paused = !simulation.Handle->paused;

			/*	simulation.Handle->OnUpdate();*/
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
		io.DisplaySize = ImVec2((float)Application::Get().GetWindow().GetWidth(), Application::Get().GetWindow().GetHeight());

		// Create an ImGui dock space
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

			ImGui::Begin("dockspace", 0, windowFlags);
			ImGui::PopStyleVar(3); // TODO: Use scoped style instead
			ImGuiID dockspaceID = ImGui::GetID("dockspace"); // TODO: Use the UI ID generator
			ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), 0);
			ImGui::End();
		}

		// Main menu bar
		{
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 0.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 2.0f, 2.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 4.0f });

			if (ImGui::BeginMainMenuBar()) {
				if (ImGui::BeginMenu("File")) {
					if (ImGui::MenuItem("Save")) {
						// Save the current scene
						const std::string filepath = m_SceneContext->GetSourceFilePath();

						if (filepath.empty()) {
							SaveCurrentSceneContext();
						}
						else {
							Application::Get().SaveCurrentSceneContext(filepath);
						}
					}

					if (ImGui::MenuItem("Save As")) {
						// Save the current scene
						SaveCurrentSceneContext();
					}

					if (ImGui::MenuItem("Open...")) {
						// Load a new scene
						LoadSceneContext();
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
						if(m_CameraControlMode == CameraControlMode::Mouse)
						{
							m_CameraControlMode = CameraControlMode::TrackPad;
						}
						else if (m_CameraControlMode == CameraControlMode::TrackPad)
						{
							m_CameraControlMode = CameraControlMode::Mouse;
						}
					}
					ImGui::EndMenu();
				}

				// TEMP
				ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 54);
				ImGui::Text("test");
				if (ImGui::Button("test")) {
					LOG("test", ConsoleColor::Green);
				}

				ImGui::EndMainMenuBar();
			}

			ImGui::PopStyleVar(3);
		}
		
		// Update panels
		m_PanelManager->OnUpdate();

		//ImGui::Begin("test");
		//UI::Image(t_Texture->GetTexture(), { (float)t_Texture->GetTexture()->GetWidth(), (float)t_Texture->GetTexture()->GetHeight() });
		//ImGui::End();


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

		// debug::Profiler::Reset();
	}
}