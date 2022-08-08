#include "pch.h"
#include "SceneHierarchyPanel.h"

#include <imgui_internal.h>
#include "UI/UI.h"
#include "Editor/Editor.h"
#include "Utility/String.h"

namespace fe {
	char SceneHierarchyPanel::s_RenameBuffer[ENTITY_NAME_MAX_LENGTH];

	SceneHierarchyPanel::SceneHierarchyPanel()
	{
		auto& assetManager = Editor::Get().GetAssetManager();
		m_TestTextureAttribute = assetManager->Add<TextureAsset>("Resources/Images/Editor/eye.png")->GetTexture();
		m_TestTextureIcon = assetManager->Get<TextureAsset>("Resources/Images/Editor/test.png")->GetTexture();
	}

	void SceneHierarchyPanel::OnUpdate()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 0.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 10.0f);
		ImGui::PushStyleColor(ImGuiCol_TableBorderLight, { 0, 0, 0, 0 });
		ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, { 0.314,0.314,0.314, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, { 0.37,0.37,0.37, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, { 0.188,0.188,0.188, 1.0f });
		ImVec2 availableSpace = ImGui::GetContentRegionAvail(); 

		// Search widget
		ImGui::SetNextItemWidth(availableSpace.x - 10.0f);
		UI::ShiftCursor(5, 3);
		UI::Widget::SearchBar(m_EntitySearchFilter, "Filter");
		UI::ShiftCursor(-5, 4);

		// List
		availableSpace = ImGui::GetContentRegionAvail();
		m_PropertiesColumnWidth = m_CurrentIconCount * (m_IconSize + m_IconSpacing) + 12;
		const ImRect windowRect = { ImGui::GetWindowContentRegionMin(), ImGui::GetWindowContentRegionMax() };
		constexpr ImGuiTableFlags tableFlags = ImGuiTableFlags_NoPadInnerX | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;

		if (ImGui::BeginTable("##SceneHierarchyTable", 2, tableFlags, availableSpace)) {
			ImGui::TableSetupColumn("##0", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, availableSpace.x - m_PropertiesColumnWidth);
			ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, m_PropertiesColumnWidth);
			UI::ListBackground();

			// Draw entity tree nodes
			for (const auto entity : m_SceneContext->m_Registry.view<IDComponent, RelationshipComponent>())
			{
				Entity e(entity, m_SceneContext.Raw());
				if (e.GetParentUUID() == 0) {
					DrawEntityNode(e, m_EntitySearchFilter);
				}
			}

			// Context menu
			{
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 8.0f, 6.0f });
				ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 2.0f);
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4, 4.0f });

				ImGui::PushStyleColor(ImGuiCol_PopupBg, (ImU32)UI::Description.ContextMenuBackground);
				ImGui::PushStyleColor(ImGuiCol_Header, (ImU32)UI::Description.Transparent);
				ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImU32)UI::Description.Transparent);
				ImGui::PushStyleColor(ImGuiCol_HeaderActive, (ImU32)UI::Description.Transparent);
				ImGui::PushStyleColor(ImGuiCol_Border, (ImU32)UI::Description.ContextMenuBorder);
				ImGui::PushStyleColor(ImGuiCol_Separator, (ImU32)UI::Description.ContextMenuBorder);

				if (ImGui::BeginPopupContextWindow(nullptr, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
					if(UI::MenuItem("Create Empty")) {
						m_SceneContext->CreateEntity("Entity");
					}

					UI::Separator();

					if (UI::MenuItem("Open Scene")) {
						Editor::Get().LoadSceneContext();
					}

					UI::ShiftCursorY(2);

					if (UI::MenuItem("Save Scene", "Ctrl + Save")) {
						Editor::Get().SaveCurrentSceneContext();
					}

					ImGui::EndPopup();
				}

				ImGui::PopStyleVar(3);
				ImGui::PopStyleColor(6);
			}
			
			ImGui::EndTable();
		}

		// BUG: When the last entity is being renamed and the user clicks away, the rename state remains active. 

		// Drag & drop
		if (ImGui::BeginDragDropTargetCustom(windowRect, ImGui::GetCurrentWindow()->ID))
		{
			const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneEntity", ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
			if (payload)
			{
				const Entity& entity = *(Entity*)payload->Data;
				m_SceneContext->UnParentEntity(entity);
				ImGui::ClearActiveID();
			}

			ImGui::EndDragDropTarget();
		}

		ImGui::PopStyleVar(2);
		ImGui::PopStyleColor(4);
	}

	bool SceneHierarchyPanel::TreeNode(Entity entity, const char* label, bool& isHovered, bool& isClicked, ImGuiID id, ImGuiTreeNodeFlags flags)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0);
		ImGui::TableNextRow(0, UI::Description.ListRowHeight);
		ImGui::TableSetColumnIndex(0);

		const ImVec2 rowAreaMin = ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), 0).Min;
		const ImVec2 rowAreaMax = { ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), ImGui::TableGetColumnCount() - 1).Max.x, rowAreaMin.y + UI::Description.ListRowHeight + 2 };

		ImGuiContext& g = *GImGui;

		// check if there are any active popups
		if (g.OpenPopupStack.Size > 0) {
			// disable hover behaviour, when a popup is active
			isHovered = false;
			isClicked = false;
		}
		else {
			ImGui::PushClipRect(rowAreaMin, rowAreaMax, false);
			isHovered = UI::ItemHoverable({ rowAreaMin, rowAreaMax }, id);
			isClicked = isHovered && ImGui::IsMouseClicked(0);
			ImGui::PopClipRect();
		}

		// Prevents rendering of items that are outside the clip rect (scroll mode)
		// Causes scroll bugs when a tree is open 
		//if (ImGui::IsClippedEx(ImRect(rowAreaMin, rowAreaMax), id)) {
		//	ImGui::PopStyleVar();
		//	return false;
		//}

		// Mouse over arrow
		auto* window = ImGui::GetCurrentWindow();
		window->DC.CurrLineSize.y = UI::Description.ListRowHeight;
		auto& style = ImGui::GetStyle();
		const char* labelEnd = ImGui::FindRenderedTextEnd(label);
		const ImVec2 padding = ((flags & ImGuiTreeNodeFlags_FramePadding)) ? style.FramePadding : ImVec2(style.FramePadding.x, ImMin(window->DC.CurrLineTextBaseOffset, style.FramePadding.y));
		const float textOffsetX = g.FontSize + padding.x * 2;
		ImVec2 textPos(window->DC.CursorPos.x + textOffsetX, window->DC.CursorPos.y + 3.0f);
		const bool isLeaf = (flags & ImGuiTreeNodeFlags_Leaf) != 0;

		if (isClicked) {
			// Mouse is hovering the arrow on the X axis && the node has children
			if ((g.IO.MousePos.x >= (textPos.x - textOffsetX) - style.TouchExtraPadding.x && g.IO.MousePos.x < (textPos.x - textOffsetX) + (g.FontSize + padding.x * 2.0f) + style.TouchExtraPadding.x) && isLeaf == false) {
				isClicked = false;
				ImGui::SetNextItemOpen(!ImGui::TreeNodeBehaviorIsOpen(id));
			}
			else {
				ImGui::SetActiveID(id, window);
			}
		}

		// Set tree node background color
		if (flags & ImGuiTreeNodeFlags_Selected) {
			ImGui::TableSetBgColor(3,UI::Description.ListSelectionActive, 0);
			ImGui::TableSetBgColor(3, UI::Description.ListSelectionActive, 1);
		}
		else if (isHovered) {
			ImGui::TableSetBgColor(3, UI::Description.ListBackgroundHovered, 0);
			ImGui::TableSetBgColor(3, UI::Description.ListBackgroundHovered, 1);
		}

		if (window->SkipItems) {
			ImGui::PopStyleVar();
			return false;
		}

		const bool isOpen = ImGui::TreeNodeBehaviorIsOpen(id, flags);
		if (isOpen && !g.NavIdIsAlive && (flags & ImGuiTreeNodeFlags_NavLeftJumpsBackHere) && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			window->DC.TreeJumpToParentOnPopMask |= (1 << window->DC.TreeDepth);
		}

		if (flags & ImGuiTreeNodeFlags_AllowItemOverlap) {
			ImGui::SetItemAllowOverlap();
		}

		// Render
		{
			// Column 0 
			// Toggle arrow
			if (!isLeaf) {
				ImGui::RenderArrow(window->DrawList, ImVec2(textPos.x - textOffsetX + padding.x, textPos.y + g.FontSize * 0.15f), UI::Description.ListToggleColor, isOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.6f);
			}

			// Icon
			UI::ShiftCursor(16, 2);
			UI::Image(m_TestTextureIcon, { (float)m_IconSize, (float)m_IconSize });

			textPos.x += 13;
			textPos.y -= 1.0f;
			if (g.LogEnabled) {
				ImGui::LogRenderedText(&textPos, ">");
			}

			ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)UI::Description.ListTextColor);

			// Rename input field
			if (m_IsRenaming && entity == m_SelectionContext) {
				UI::ShiftCursor(32, -21);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x); // TODO: check what looks / works better
				ImGui::SetKeyboardFocusHere();

				ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);
				ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImU32)UI::Description.InputFieldBackground);
				ImGui::InputTextWithHint("##rename", entity.GetComponent<TagComponent>().Tag.c_str(), s_RenameBuffer, ENTITY_NAME_MAX_LENGTH);
				ImGui::PopStyleColor();
				ImGui::PopStyleVar();

				if (ImGui::IsItemDeactivatedAfterEdit())
				{
					RenameEntity();
				}
			}
			// Node label
			else {
				ImGui::RenderText(textPos, label, labelEnd, false);
			}
			ImGui::PopStyleColor();

			// Column 1
			// Draw entity components icons here
			ImGui::TableSetColumnIndex(1);
			UI::ShiftCursor(4.0f, 2.0f);
			ImVec2 cursorPos = ImGui::GetCursorPos();

			cursorPos.x += m_PropertiesColumnWidth - m_IconSize - 17;
			ImGui::SetCursorPos(cursorPos);

			for (size_t i = 0; i < m_CurrentIconCount; i++)
			{
				UI::Image(m_TestTextureAttribute, { (float)m_IconSize, (float)m_IconSize });
				cursorPos.x -= m_IconSize + m_IconSpacing;
				ImGui::SetCursorPos(cursorPos);
			}
		}

		ImGui::PushClipRect(rowAreaMin, rowAreaMax, false);
		ImGui::ItemAdd(ImRect(rowAreaMin, rowAreaMax), id);
		ImGui::PopClipRect();

		if (isOpen && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			ImGui::TreePushOverrideID(id);
		}

		IMGUI_TEST_ENGINE_ITEM_INFO(id, label, window->DC.ItemFlags | (isLeaf ? 0 : ImGuiItemStatusFlags_Openable) | (isOpen ? ImGuiItemStatusFlags_Opened : 0));
		ImGui::PopStyleVar();

		if (isHovered && ImGui::IsMouseReleased(0)) {
			ImGui::ClearActiveID();
		}

		return isOpen;
	}

	void SceneHierarchyPanel::DrawEntityNode(Entity entity, const std::string& filter)
	{
		const char* name = entity.GetComponent<TagComponent>().Tag.c_str();

		constexpr  uint32_t maxSearchDepth = 10;
		const bool hasChildMatchingSearch = TagSearchRecursive(entity, filter, maxSearchDepth);

		if (!IsMatchingSearch(name, filter) && !hasChildMatchingSearch) {
			return;
		}

		const bool isSelected = entity == m_SelectionContext;
		bool isDeleted = false;
		const std::string strID = std::to_string((uint32_t)entity);
		ImGuiTreeNodeFlags flags = (isSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth;

		if (entity.Children().empty()) {
			flags |= ImGuiTreeNodeFlags_Leaf;
		}

		bool hovered;
		bool clicked;
		bool doubleClicked;

		const ImGuiID treeNodeId = ImGui::GetID(strID.c_str());

		// Draw tree node
		bool opened = TreeNode(entity, name, hovered, clicked, treeNodeId, flags);

		if (clicked)
		{
			// Submit the new name upon clicking on any node
			if (m_IsRenaming) {
				RenameEntity();
			}

			Editor::Get().SetSelectionContext(entity);
		}

		// Context menu
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 8.0f, 6.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 2.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4, 4.0f });

			ImGui::PushStyleColor(ImGuiCol_PopupBg, (ImU32)UI::Description.ContextMenuBackground);
			ImGui::PushStyleColor(ImGuiCol_Header, (ImU32)UI::Description.Transparent);
			ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImU32)UI::Description.Transparent);
			ImGui::PushStyleColor(ImGuiCol_HeaderActive, (ImU32)UI::Description.Transparent);
			ImGui::PushStyleColor(ImGuiCol_Border, (ImU32)UI::Description.ContextMenuBorder);
			ImGui::PushStyleColor(ImGuiCol_Separator, (ImU32)UI::Description.ContextMenuBorder);

			if (ImGui::BeginPopupContextItem()) {
				if (UI::MenuItem("Create Empty")) {
					m_SceneContext->CreateChildEntity(entity, "Entity");
					opened = true;
				}

				UI::Separator();

				if (UI::MenuItem("Delete", "Delete")) {
					isDeleted = true;
				}

				UI::ShiftCursorY(2);

				if (UI::MenuItem("Rename")) {
					m_IsRenaming = true;
					Editor::Get().SetSelectionContext(entity);
				}

				UI::Separator();

				if (UI::BeginMenu("Add Component")) {
					UI::MenuItem("Material");
					UI::ShiftCursorY(2);
					UI::MenuItem("Mesh");
					UI::ShiftCursorY(2);
					UI::MenuItem("SPH Simulation");

					ImGui::EndMenu();
				}

				ImGui::EndPopup();
			}

			ImGui::PopStyleVar(3);
			ImGui::PopStyleColor(6);
		}

		// Drag & drop
		{
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
			{
				ImGui::Text(name);
				ImGui::SetDragDropPayload("SceneEntity", &entity, sizeof(Entity));
				ImGui::EndDragDropSource();
			}

			if (ImGui::BeginDragDropTarget())
			{
				const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneEntity", ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
				if (payload)
				{
					const Entity& droppedEntity = *(Entity*)payload->Data;
					m_SceneContext->ParentEntity(droppedEntity, entity);
				}

				ImGui::EndDragDropTarget();
			}
		}

		// Draw child nodes
		if (opened)
		{
			for (const auto child : entity.Children()) {
				DrawEntityNode(m_SceneContext->GetEntityWithUUID(child), filter);
			}

			ImGui::TreePop();
		}

		// Defer deletion until the end node drawing.
		if (isDeleted) {
			if (entity == m_SelectionContext) {
				Editor::Get().SetSelectionContext({});
			}

			m_SceneContext->DeleteEntity(entity);
		}
	}

	bool SceneHierarchyPanel::TagSearchRecursive(Entity entity, std::string_view searchFilter, uint32_t maxSearchDepth, uint32_t currentDepth)
	{
		if (searchFilter.empty()) {
			return false;
		}

		for (const auto child : entity.Children())
		{
			Entity e = m_SceneContext->GetEntityWithUUID(child);
			if (e.HasComponent<TagComponent>())
			{
				if (IsMatchingSearch(e.GetComponent<TagComponent>().Tag, searchFilter)) {
					return true;
				}
			}

			if (TagSearchRecursive(e, searchFilter, maxSearchDepth, currentDepth + 1))
			{
				return true;
			}
		}
		return false;
	}

	bool SceneHierarchyPanel::IsMatchingSearch(const std::string& item, const std::string_view searchQuery,const bool caseSensitive,const bool stripWhiteSpaces,const bool stripUnderscores)
	{
		if (searchQuery.empty()) {
			return true;
		}

		if (item.empty()) {
			return false;
		}

		std::string itemSanitized = stripUnderscores ? Replace(item, "_", " ") : item;

		if (stripWhiteSpaces) {
			itemSanitized = Replace(itemSanitized, " ", "");
		}

		std::string searchString = stripWhiteSpaces ? Replace(searchQuery, " ", "") : std::string(searchQuery);

		if (caseSensitive == false)
		{
			itemSanitized = ToLower(itemSanitized);
			searchString = ToLower(searchString);
		}

		bool result = false;
		if (Contains(searchString, " "))
		{
			std::vector<std::string> searchTerms;
			Split(searchString, " ", searchTerms);
			for (const auto& searchTerm : searchTerms)
			{
				if (!searchTerm.empty() && Contains(itemSanitized, searchTerm)) {
					result = true;
				}
				else
				{
					result = false;
					break;
				}
			}
		}
		else
		{
			result = Contains(itemSanitized, searchString);
		}

		return result;
	}
	
	void SceneHierarchyPanel::RenameEntity()
	{
		if (m_SelectionContext) {
			// Don't rename the entity if the name buffer isn't empty
			if (strlen(s_RenameBuffer) != 0) {
				m_SelectionContext.GetComponent<TagComponent>().Tag = s_RenameBuffer;
			}
		}

		ClearRenameBuffer();
		m_IsRenaming = false;
	}

	void SceneHierarchyPanel::ClearRenameBuffer()
	{
		memset(s_RenameBuffer, 0, ENTITY_NAME_MAX_LENGTH);
	}
}