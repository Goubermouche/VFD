#ifndef PANEL_MANAGER_H_
#define PANEL_MANAGER_H_

#include "FluidEngine/Editor/Panels/EditorPanel.h"
#include "FluidEngine/Core/Cryptography/Hash.h"
#include "FluidEngine/Scene/Scene.h"

namespace fe {
	class PanelManager
	{
	public:
		template<typename TPanel, typename... TArgs>
		Ref<TPanel> AddPanel(const std::string& ID, TArgs&&... args)
		{
			static_assert(std::is_base_of<EditorPanel, TPanel>::value, "panel does not inherit from EditorPanel!");

			uint32_t IDHash = Hash::GenerateFNVHash(ID) + m_Panels.size();
			Ref<TPanel> panel = Ref<TPanel>::Create(std::forward<TArgs>(args)...);
			panel->m_Name = ID + "##" + std::to_string(IDHash);
			m_Panels[IDHash] = panel;

			return panel;
		}

		void OnUpdate() {
			for (auto& [id, panelData] : m_Panels)
			{
				panelData->OnUpdate();
			}
		}

		void OnEvent(Event& e)
		{
			for (auto& [id, panelData] : m_Panels) {
				panelData->OnEvent(e);
			}
		}

		void OnSceneContextChanged(Ref<Scene> context) {
			for (auto& [id, panelData] : m_Panels) {
				panelData->SetSceneContext(context);
			}
		}
	private:
		std::unordered_map<uint32_t, Ref<EditorPanel>> m_Panels;
	};
}

#endif // !PANEL_MANAGER_H_
