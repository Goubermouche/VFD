#include "pch.h"
#include "SceneHierarchyPanel.h"

namespace fe {
	SceneHierarchyPanel::SceneHierarchyPanel()
	{

	}

	void SceneHierarchyPanel::OnUpdate()
	{
		ImGui::Begin(m_Name.c_str());
		m_Hovered = ImGui::IsWindowHovered();

		ImGui::Text(std::to_string(m_SceneContext->GetEntityCount()).c_str());
		ImGui::End();
	}

	void SceneHierarchyPanel::OnEvent(Event& e)
	{

	}

	void SceneHierarchyPanel::SetSceneContext(Ref<Scene> context)
	{
		m_SceneContext = context;
	}
}