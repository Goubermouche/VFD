#include "pch.h"
#include "PanelManager.h"

namespace fe {
	bool PanelManager::OnMousePress(MouseButtonPressedEvent& event)
	{
		for (auto& [id, panel] : m_Panels) {
			panel->m_Focused = panel->m_Hovered;
		}
		return false;
	}

	bool PanelManager::OnMouseScroll(MouseScrolledEvent& event)
	{
		for (auto& [id, panel] : m_Panels) {
			panel->m_Focused = panel->m_Hovered;
		}
		return false;
	}
}
