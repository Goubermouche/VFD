#ifndef SCENE_HIERARCHY_PANEL_H_
#define SCENE_HIERARCHY_PANEL_H_

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

		virtual void OnUpdate() override;
	private:
		void DrawEntityNode(Entity entity);
		bool DrawTreeNode(const char* label, bool* outHovered, bool* outClicked, ImGuiID id, ImGuiTreeNodeFlags flags);
	};
}

#endif // !SCENE_HIERARCHY_PANEL_H_