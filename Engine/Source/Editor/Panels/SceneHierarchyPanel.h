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
		void OnEvent(Event& event) override;
	private:
		bool TreeNode(Entity entity, const char* label, bool& hovered, bool& clicked, ImGuiID id, ImGuiTreeNodeFlags flags);
		void DrawEntityNode(Entity entity, const std::string& filter);
		bool TagSearchRecursive(Entity entity, std::string_view searchFilter, uint32_t maxSearchDepth, uint32_t currentDepth = 1);
		static bool IsMatchingSearch(const std::string& item, std::string_view searchQuery, bool caseSensitive = false, bool stripWhiteSpaces = false, bool stripUnderscores = false);

		void RenameEntity();
		static void ClearRenameBuffer();

		bool OnKeyPressed(KeyPressedEvent& event);
	private:
		Ref<Texture> m_FileTexture;
		Ref<Texture> m_FolderTexture;

		// Entity renaming
		Entity m_RenameContext;
		static char s_RenameBuffer[ENTITY_NAME_MAX_LENGTH];

		float m_PropertiesColumnWidth;
		std::string m_EntitySearchFilter;
	};
}

#endif // !SCENE_HIERARCHY_PANEL_H