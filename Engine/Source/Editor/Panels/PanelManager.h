#ifndef PANEL_MANAGER_H
#define PANEL_MANAGER_H

#include "Editor/Panels/EditorPanel.h"
#include "Core/Cryptography/Hash.h"

namespace fe {
	class Scene;
	class Entity;
	class EditorPanel;

	/// <summary>
	/// Manages editor panels. There should be 0 interaction between the editor itself and panels.
	/// </summary>
	class PanelManager : public RefCounted
	{
	public:
		PanelManager() = default;
		~PanelManager() = default;

		template<typename TPanel, typename... TArgs>
		Ref<TPanel> AddPanel(const std::string& name, TArgs&&... args);

		void OnUpdate();
		void OnEvent(Event& event);
		
		void SetSceneContext(Ref<Scene> context);
		void SetSelectionContext(Entity context);

	private:
		bool OnMousePress(MouseButtonPressedEvent& event);
		bool OnMouseScroll(MouseScrolledEvent& event);

	private:
		std::unordered_map<uint32_t, Ref<EditorPanel>> m_Panels;
	};

	template<typename TPanel, typename ...TArgs>
	inline Ref<TPanel> PanelManager::AddPanel(const std::string& name, TArgs && ...args)
	{
		static_assert(std::is_base_of<EditorPanel, TPanel>::value, "panel does not inherit from EditorPanel!");

		const uint32_t IDHash = Hash::GenerateFNVHash(name) + m_Panels.size();
		Ref<TPanel> panel = Ref<TPanel>::Create(std::forward<TArgs>(args)...);
		panel->m_ID = name + "##" + std::to_string(IDHash);
		m_Panels[IDHash] = panel;

		return panel;
	}
}

#endif // !PANEL_MANAGER_H
