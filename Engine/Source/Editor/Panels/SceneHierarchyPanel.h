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
		~SceneHierarchyPanel() override = default;

		void OnUpdate() override;
	private:
		bool TreeNode(Entity entity, const char* label, bool& isHovered, bool& isClicked, ImGuiID id, ImGuiTreeNodeFlags flags);
		void DrawEntityNode(Entity entity, const std::string& filter);
		bool TagSearchRecursive(Entity entity, std::string_view searchFilter, uint32_t maxSearchDepth, uint32_t currentDepth = 1);
		static bool IsMatchingSearch(const std::string& item, std::string_view searchQuery, bool caseSensitive = false, bool stripWhiteSpaces = false, bool stripUnderscores = false);

		void RenameEntity();
		void ClearRenameBuffer();
	private:
		Ref<Texture> m_TestTextureAttribute;
		Ref<Texture> m_TestTextureIcon;

		const uint32_t m_IconSize = 18;
		const uint32_t m_IconSpacing = 2;
		uint32_t m_CurrentIconCount = 2;

		float m_PropertiesColumnWidth;

		bool m_IsRenaming = false;
		static char s_RenameBuffer[255];
	};
}

#endif // !SCENE_HIERARCHY_PANEL_H