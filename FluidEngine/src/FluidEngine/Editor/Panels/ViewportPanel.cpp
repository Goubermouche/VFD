#include "pch.h"
#include "ViewportPanel.h"

namespace fe {
	ViewportPanel::ViewportPanel()
	{
	}

	void ViewportPanel::OnUpdate()
	{
		ImGui::Begin(m_Name.c_str());
		// Maybe replace the ImGui::Begin() and ImGui::End() calls with a function inside the editor panel and handle the hover event there? 
		m_Hovered = ImGui::IsWindowHovered();

		ImGui::End();
	}

	void ViewportPanel::OnEvent(Event& e)
	{
	}

	void ViewportPanel::SetSceneContext(Ref<Scene> context)
	{
		m_SceneContext = context;
	}
}