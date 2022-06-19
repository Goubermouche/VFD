#ifndef PANEL_H_
#define PANEL_H_

#include "pch.h"
#include "imgui.h"
#include "FluidEngine/Scene/Scene.h"
#include "FluidEngine/Scene/Entity.h"

namespace fe {
	/// <summary>
	/// Base editor panel class. 
	/// </summary>
	class EditorPanel : public RefCounted {
	public:
		virtual void OnUpdate() = 0;
		virtual void OnEvent(Event& e) {};

		virtual void SetSceneContext(Ref<Scene> context) {
			m_SceneContext = context;
		};

		virtual void SetSelectionContext(Entity context) {
			m_SelectionContext = context;
		};
	protected:
		std::string m_ID;
		Ref<Scene> m_SceneContext;
		Entity m_SelectionContext;

		bool m_Focused = false;
		bool m_Hovered = false;

		friend class PanelManager;
	};
}

#endif // !PANEL_H_