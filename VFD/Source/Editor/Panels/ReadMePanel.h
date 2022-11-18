#ifndef README_PANEL_H
#define README_PANEL_H

#include "Editor/Panels/EditorPanel.h"

namespace vfd {
	class EditorCamera;

	class ReadMePanel : public EditorPanel
	{
	public:
		ReadMePanel();
		~ReadMePanel() override = default;

		void OnUpdate() override;
		void OnEvent(Event& event) override;
	private:
		bool OnSceneSaved(SceneSavedEvent& event);
		bool OnSceneLoaded(SceneLoadedEvent& event);
	private:
		std::string m_ReadmeText;
	};
}

#endif // !README_PANEL_H