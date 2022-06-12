#include "pch.h"
#include "SceneHierarchyPanel.h"

namespace fe {
	SceneHierarchyPanel::SceneHierarchyPanel()
	{
	}

	void SceneHierarchyPanel::OnUpdate()
	{
		ImGui::Begin(m_Name.c_str());
		ImGui::End();
	}

	void SceneHierarchyPanel::OnEvent(Event& e)
	{
	}
}
