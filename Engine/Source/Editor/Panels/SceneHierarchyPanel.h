#ifndef SCENE_HIERARCHY_PANEL_H
#define SCENE_HIERARCHY_PANEL_H

#include "Editor/Panels/EditorPanel.h"
#include "Scene/Entity.h"

namespace fe {
	/// <summary>
	/// Scene hierarchy panel. Displays all entities in the current scene.
	/// </summary>
	class SceneHierarchyPanel : public EditorPanel
	{
	public:
		SceneHierarchyPanel();

		void OnUpdate() override;
	private:
		void DrawEntityNode(Entity entity);
		bool DrawTreeNode(const char* label, bool& isHovered, bool& isClicked, ImGuiID id, ImGuiTreeNodeFlags flags) const;
	};
}

#endif // !SCENE_HIERARCHY_PANEL_H