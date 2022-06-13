#include "pch.h"
#include "Editor.h"

#include "FluidEngine/Platform/ImGui/ImGuiRenderer.h" 
#include "FluidEngine/Platform/ImGui/ImGuiGLFWBackend.h"
#include "FluidEngine/Core/Application.h"

namespace fe {
	Editor::Editor()
	{
		// Init ImGui
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow()), true);
		ImGui_ImplOpenGL3_Init("#version 410");

		m_PanelManager.reset(new PanelManager());
		m_PanelManager->AddPanel<SceneHierarchyPanel>("Scene");
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

	void Editor::Ondate()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGuiIO& io = ImGui::GetIO();
		Application& app = Application::Get();
		io.DisplaySize = ImVec2((float)Application::Get().GetWindow().GetWidth(), (float)Application::Get().GetWindow().GetHeight());

		// Update panels
		m_PanelManager->OnUpdate();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}