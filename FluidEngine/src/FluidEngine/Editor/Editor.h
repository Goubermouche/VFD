#ifndef EDITOR_H_
#define EDITOR_H_

#include "Panels/PanelManager.h"

// Panels
#include "FluidEngine/Editor/Panels/SceneHierarchyPanel.h"
#include "FluidEngine/Editor/Panels/ViewportPanel.h"

namespace fe {
	class Editor : public RefCounted {
	public:
		Editor();
		~Editor();

		void OnUpdate();
		void OnEvent(Event& event);
		void OnSceneContextChanged(Ref<Scene> context);
	private:
		void InitImGui();

		// Events
		bool OnKeyPressed(KeyPressedEvent& e);
	private:
		std::unique_ptr<PanelManager> m_PanelManager;
	};
}

#endif // !EDITOR_H_