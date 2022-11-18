#include "pch.h"
#include "PanelManager.h"

namespace vfd {
	void PanelManager::OnUpdate()
	{
		for (auto& [id, panel] : m_Panels) {
			// Handle ImGui windows here.
			if (panel->m_Enabled) {
				if (ImGui::Begin(panel->m_ID.c_str())) {
					panel->m_Hovered = ImGui::IsWindowHovered();
					panel->OnUpdate();
				}
				ImGui::End();
			}
		}
	}

	void PanelManager::OnEvent(Event& event)
	{
		// Dispatch window focus events 
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<MouseButtonPressedEvent>(BIND_EVENT_FN(OnMousePress));
		dispatcher.Dispatch<MouseScrolledEvent>(BIND_EVENT_FN(OnMouseScroll));

		// Bubble unhandled events further
		if (event.handled == false) {
			for (auto& [id, panel] : m_Panels) {
				panel->OnEvent(event);
			}
		}
	}

	void PanelManager::SetSceneContext(Ref<Scene> context)
	{
		for (auto& [id, panel] : m_Panels) {
			panel->SetSceneContext(context);
		}
	}

	void PanelManager::SetSelectionContext(Entity context)
	{
		for (auto& [id, panel] : m_Panels) {
			panel->SetSelectionContext(context);
		}
	}

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
