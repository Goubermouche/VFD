#ifndef TIMELINE_PANEL_H
#define TIMELINE_PANEL_H

#include "Editor/Panels/EditorPanel.h"

namespace vfd {
	class TimelinePanel : public EditorPanel
	{
	public:
		TimelinePanel();
		~TimelinePanel() override = default;

		void OnUpdate() override;
		void OnEvent(Event& event) override;
	private:
		bool OnKeyPressed(KeyPressedEvent& event);
	private:
		float m_CursorPositionTimeline = 0.0f;
		float m_FrameIndex = 0.0f;

		bool m_Paused = true;
	};
}

#endif // !TIMELINE_PANEL_H