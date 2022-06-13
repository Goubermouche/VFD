#ifndef EDITOR_H_
#define EDITOR_H_

#include "FluidEngine/Editor/Panels/SceneHierarchyPanel.h"
#include "Panels/PanelManager.h"

namespace fe {
	class Editor : public RefCounted {
	public:
		Editor();
		~Editor();

		void Ondate();
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