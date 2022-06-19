#ifndef SCENE_HIERARCHY_PANEL_H_
#define SCENE_HIERARCHY_PANEL_H_

#include "FluidEngine/Editor/Panels/EditorPanel.h"
#include "FluidEngine/Scene/Entity.h"

namespace fe {
	/// <summary>
	/// Scene hierarchy panel. Displays all entities in the current scene.
	/// </summary>
	class SceneHierarchyPanel : public EditorPanel
	{
	public:
		SceneHierarchyPanel();

		virtual void OnUpdate() override;
		virtual void OnEvent(Event& e) override;

		virtual void SetSceneContext(Ref<Scene> context) override;
		virtual void SetSelectionContext(Entity selectionContext) override;
	private:
		void DrawEntityNode(Entity entity);
		bool DrawTreeNode(const char* label, bool* outHovered, bool* outClicked, ImGuiID id, ImGuiTreeNodeFlags flags);
	};
}

#endif // !SCENE_HIERARCHY_PANEL_H_