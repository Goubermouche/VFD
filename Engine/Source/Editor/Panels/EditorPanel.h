#ifndef PANEL_H
#define PANEL_H

#include "pch.h"
#include "imgui.h"
#include "Scene/Scene.h"
#include "Scene/Entity.h"

namespace fe {
	/// <summary>
	/// Base editor panel class. 
	/// </summary>
	class EditorPanel : public RefCounted {
	public:
		virtual void OnUpdate() = 0;
		virtual void OnEvent(Event& event)
		{}
		virtual ~EditorPanel() = default;

		virtual void SetSceneContext(const Ref<Scene> context) {
			m_SceneContext = context;
		};

		virtual void SetSelectionContext(const Entity context) {
			m_SelectionContext = context;
		};

		void SetEnabled(bool state) {
			m_Enabled = state;
		}
	protected:
		std::string m_ID; // ImGui id of the panel
		Ref<Scene> m_SceneContext;
		Entity m_SelectionContext;

		bool m_Focused = false;
		bool m_Hovered = false;
		bool m_Enabled = true;

		friend class PanelManager;
	};
}

#endif // !PANEL_H