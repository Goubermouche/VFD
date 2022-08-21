#include "pch.h"
#include "ReadMePanel.h"

namespace fe {
	ReadMePanel::ReadMePanel()
	{
	}

	void ReadMePanel::OnUpdate()
	{
		ImGui::TextWrapped(m_ReadmeText.c_str());
	}

	void ReadMePanel::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<SceneSavedEvent>(BIND_EVENT_FN(OnSceneSaved));
		dispatcher.Dispatch<SceneLoadedEvent>(BIND_EVENT_FN(OnSceneLoaded));
	}

	bool ReadMePanel::OnSceneSaved(SceneSavedEvent& event)
	{
		return false;
	}

	bool ReadMePanel::OnSceneLoaded(SceneLoadedEvent& event)
	{
		m_ReadmeText = m_SceneContext->GetData().ReadMe;
		return false;
	}
}