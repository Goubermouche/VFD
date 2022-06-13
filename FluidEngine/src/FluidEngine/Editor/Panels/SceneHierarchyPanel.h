#ifndef SCENE_HIERARCHY_PANEL_H_
#define SCENE_HIERARCHY_PANEL_H_

#include "FluidEngine/Editor/Panels/EditorPanel.h"

namespace fe {
	class SceneHierarchyPanel : public EditorPanel
	{
	public:
		SceneHierarchyPanel();

		virtual void OnUpdate() override;
		virtual void OnEvent(Event& e) override;
		virtual void SetSceneContext(Ref<Scene> context) override;
	private:
	};
}

#endif // !SCENE_HIERARCHY_PANEL_H_