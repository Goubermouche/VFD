#ifndef PANEL_MANAGER_H
#define PANEL_MANAGER_H

#include "Editor/Panels/EditorPanel.h"
#include "Core/Cryptography/Hash.h"
#include "Scene/Scene.h"
#include "Scene/Entity.h"

namespace fe {
	/// <summary>
	/// Manages editor panels. There should be 0 interaction between the editor itself and panels.
	/// </summary>
	class PanelManager
	{
	public:
		PanelManager() = default;
		~PanelManager() = default;

		template<typename TPanel, typename... TArgs>
		Ref<TPanel> AddPanel(const std::string& ID, TArgs&&... args)
		{
			static_assert(std::is_base_of<EditorPanel, TPanel>::value, "panel does not inherit from EditorPanel!");

			const uint32_t IDHash = Hash::GenerateFNVHash(ID) + m_Panels.size();
			Ref<TPanel> panel = Ref<TPanel>::Create(std::forward<TArgs>(args)...);
			panel->m_ID = ID + "##" + std::to_string(IDHash);
			m_Panels[IDHash] = panel;

			return panel;
		}

		void OnUpdate() {
			for (auto& [id, panel] : m_Panels){
				// Handle ImGui windows here.
				if (ImGui::Begin(panel->m_ID.c_str(), 0, panel->m_Flags)) {
					panel->m_Hovered = ImGui::IsWindowHovered();
					panel->OnUpdate();
				}
				ImGui::End();
			}
		}

		void OnEvent(Event& event)
		{
			// Dispatch window focus events 
			EventDispatcher dispatcher(event);
			dispatcher.Dispatch<MouseButtonPressedEvent>([this](MouseButtonPressedEvent& e) {
				return OnMousePress(e);
			});

			dispatcher.Dispatch<MouseScrolledEvent>([this](MouseScrolledEvent& e) {
				return OnMouseScroll(e);
			});

			// Bubble unhandled events further
			if (event.handled == false) {
				for (auto& [id, panel] : m_Panels) {
					panel->OnEvent(event);
				}
			}
		}

		void SetSceneContext(Ref<Scene> context) {
			for (auto& [id, panel] : m_Panels) {
				panel->SetSceneContext(context);
			}
		}

		void SetSelectionContext(Entity context) {
			for (auto& [id, panel] : m_Panels) {
				panel->SetSelectionContext(context);
			}
		}
	private:
		bool OnMousePress(MouseButtonPressedEvent& e);
		bool OnMouseScroll(MouseScrolledEvent& e);
	private:
		std::unordered_map<uint32_t, Ref<EditorPanel>> m_Panels;
	};
}

#endif // !PANEL_MANAGER_H
