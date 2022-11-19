#include "pch.h"
#include "Editor.h"

// Panels
#include "Editor/Panels/SceneHierarchyPanel.h"
#include "Editor/Panels/ViewportPanel.h"
#include "Editor/Panels/ProfilerPanel.h"
#include "Editor/Panels/ComponentPanel.h"

#include "UI/ImGui/ImGuiRenderer.h" 
#include "UI/ImGui/ImGuiGLFWBackend.h"

#include "Utility/FileSystem.h"
#include "UI/UI.h"

namespace vfd {
	Editor* Editor::s_Instance = nullptr;
	
	Editor::Editor()
	{
		s_Instance = this;;

		// Init the asset manager
		m_AssetManager = Ref<AssetManager>::Create();
		m_AssetManager->GetOrCreateAsset<TextureAsset>("Resources/Images/Editor/file.png");
		m_AssetManager->GetOrCreateAsset<TextureAsset>("Resources/Images/Editor/folder.png");
		
		// Init the UI
		UI::Init();

		// Init panels
		m_PanelManager = Ref<PanelManager>::Create();
		m_PanelManager->AddPanel<ViewportPanel>("Viewport");
	 	m_PanelManager->AddPanel<ProfilerPanel>("Profiler");
		m_PanelManager->AddPanel<SceneHierarchyPanel>("Scene");
		m_PanelManager->AddPanel<ComponentPanel>("Components");
		m_ReadMePanel = m_PanelManager->AddPanel<ReadMePanel>("Notes");

		m_ReadMePanel->SetEnabled(false);
	}

	void Editor::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<KeyPressedEvent>(BIND_EVENT_FN(OnKeyPressed));
		dispatcher.Dispatch<SceneSavedEvent>(BIND_EVENT_FN(OnSceneSaved));
		dispatcher.Dispatch<SceneLoadedEvent>(BIND_EVENT_FN(OnSceneLoaded));

		// Pass unhandled events down to panels
		if (event.handled == false) {
			m_PanelManager->OnEvent(event);
		}
	}

	void Editor::SetSceneContext(const Ref<Scene> context)
	{
		m_SceneContext = context;
		m_PanelManager->SetSceneContext(context);
		SetSelectionContext({});
	}

	void Editor::SetSelectionContext(const Entity context)
	{
		m_SelectionContext = context;
		m_PanelManager->SetSelectionContext(m_SelectionContext);
	}

	Entity Editor::GetSelectionContext()
	{
		return m_SelectionContext;
	}

	void Editor::SaveCurrentSceneContext()
	{
		// Open a save file dialog.
		const std::string filepath = fs::FileDialog::SaveFile("Json files (*.json)|*.json|Text files (*.txt)|*.txt", "json");
		if (filepath.empty() == false) {
			Application::Get().DispatchEvent<SceneSavedEvent, true>();
			m_SceneContext->Save(filepath);
		}
	}

	void Editor::LoadSceneContext()
	{
		// Open an open file dialog.
		const std::string filepath = fs::FileDialog::OpenFile("Json files (*.json)|*.json|Text files (*.txt)|*.txt");
		if (filepath.empty() == false) {
			// Call the application save API.
			m_SceneContext->Load(filepath);
			Application::Get().DispatchEvent<SceneLoadedEvent, true>();
		}
	}

	CameraControlMode Editor::GetCameraMode() const
	{
		return m_CameraControlMode;
	}

	Editor& Editor::Get()
	{
		return *s_Instance;
	}

	Ref<AssetManager>& Editor::GetAssetManager()
	{
		return m_AssetManager;
	}

	// Global editor shortcuts should be placed here.
	// TODO: Potentially replace these if statements by a shortcut data structure.
	bool Editor::OnKeyPressed(KeyPressedEvent& event)
	{
		// Call shortcuts only when the key get pressed.
		if (event.GetRepeatCount() > 0) {
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
				const std::string filepath = m_SceneContext->GetSourceFilepath();

				if (filepath.empty()) {
					SaveCurrentSceneContext();
				}
				else {
					Application::Get().DispatchEvent<SceneSavedEvent, true>();
					m_SceneContext->Save(filepath);
				}
			}
		}

		// Pause/Unpause all currently running simulations in the scene
		if (Input::IsKeyPressed(KEY_SPACE)) {
			for (const entt::entity entity : m_SceneContext->View<SPHSimulationComponent>()) {
				Entity e = { entity, m_SceneContext.Raw()};
				auto& simulation = e.GetComponent<SPHSimulationComponent>();

				simulation.Handle->paused = !simulation.Handle->paused;
			}

			for (const entt::entity entity : m_SceneContext->View<DFSPHSimulationComponent>()) {
				Entity e = { entity, m_SceneContext.Raw() };
				auto& simulation = e.GetComponent<DFSPHSimulationComponent>();

				simulation.Handle->paused = !simulation.Handle->paused;
			}
		}

		if (Input::IsKeyPressed(KEY_R)) {
			for (const entt::entity entity : m_SceneContext->View<SPHSimulationComponent>()) {
				Entity e = { entity, m_SceneContext.Raw() };
				auto& simulation = e.GetComponent<SPHSimulationComponent>();

				simulation.Handle->Reset();
			}
		}

		return false;
	}

	bool Editor::OnSceneSaved(SceneSavedEvent& event)
	{
		return false;
	}

	bool Editor::OnSceneLoaded(SceneLoadedEvent& event)
	{
		SceneData& data = m_SceneContext->GetData(); 

		// Only enable the ReadMe panel when there is something to display
		m_ReadMePanel->SetEnabled(data.ReadMe.empty() == false);

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

		// Update panels
		m_PanelManager->OnUpdate();

		// ImGui::ShowStyleEditor();
		// ImGui::ShowDemoWindow();

		// End ImGui frame
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}