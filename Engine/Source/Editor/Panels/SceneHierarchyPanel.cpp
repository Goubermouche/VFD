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
		m_FileTexture = assetManager->Get<TextureAsset>("Resources/Images/Editor/file.png")->GetTexture();
		m_FolderTexture = assetManager->Get<TextureAsset>("Resources/Images/Editor/folder.png")->GetTexture();
	}

	void SceneHierarchyPanel::OnUpdate()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 4.0f, 0.0f });
		ImGui::PushStyleColor(ImGuiCol_TableBorderLight, { 0.0f, 0.0f, 0.0f, 0.0f });
		ImVec2 availableSpace = ImGui::GetContentRegionAvail();

		// Search widget
		//ImGui::SetNextItemWidth(availableSpace.x - 10.0f);
		//UI::ShiftCursor(5, 3);
		//UI::Widget::SearchBar(m_EntitySearchFilter, "Filter");
		//UI::ShiftCursor(-5, 4);

		// List
		availableSpace = ImGui::GetContentRegionAvail();
		m_PropertiesColumnWidth = 40;
		const ImRect windowRect = { ImGui::GetWindowContentRegionMin(), ImGui::GetWindowContentRegionMax() };
		constexpr ImGuiTableFlags tableFlags = ImGuiTableFlags_NoPadInnerX | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;

		if (ImGui::BeginTable("##SceneHierarchyTable", 2, tableFlags, availableSpace)) {
			ImGui::TableSetupColumn("##0", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, availableSpace.x - m_PropertiesColumnWidth);
			ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, m_PropertiesColumnWidth);

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
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 4.0f });

				if (ImGui::BeginPopupContextWindow(nullptr, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
					if (ImGui::MenuItem("Create Empty")) {
						m_SceneContext->CreateEntity("Entity");
					}

					ImGui::Separator();

					if (ImGui::MenuItem("Open Scene")) {
						Editor::Get().LoadSceneContext();
					}

					if (ImGui::MenuItem("Save Scene", "Ctrl S")) {
						Editor::Get().SaveCurrentSceneContext();
					}

					ImGui::EndPopup();
				}

				ImGui::PopStyleVar();
			}

			ImGui::EndTable();
		}

		if (ImGui::IsMouseClicked(0 && m_RenameContext)) {
			RenameEntity();
		}

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

		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
	}

	void SceneHierarchyPanel::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<KeyPressedEvent>(BIND_EVENT_FN(OnKeyPressed));
	}

	void SceneHierarchyPanel::DrawEntityNode(Entity entity, const std::string& filter)
	{
		const char* name = entity.GetComponent<TagComponent>().Tag.c_str();

		constexpr uint32_t maxSearchDepth = 10;
		const bool hasChildMatchingSearch = TagSearchRecursive(entity, filter, maxSearchDepth);

		if (IsMatchingSearch(name, filter) == false && hasChildMatchingSearch == false) {
			return;
		}

		const bool selected = entity == m_SelectionContext;
		bool deleted = false;
		const std::string stringID = std::to_string((uint32_t)entity);
		ImGuiTreeNodeFlags flags = (selected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth;

		if (entity.Children().empty()) {
			flags |= ImGuiTreeNodeFlags_Leaf;
		}

		bool hovered;
		bool clicked;
		bool doubleClicked;

		const ImGuiID treeNodeId = ImGui::GetID(stringID.c_str());

		// Draw tree node
		bool opened = TreeNode(entity, name, hovered, clicked, treeNodeId, flags);

		if (clicked) {
			Editor::Get().SetSelectionContext(entity);
		}

		// Context menu
		{
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 4.0f });

			if (ImGui::BeginPopupContextItem()) {
				if (ImGui::MenuItem("Create Empty")) {
					m_SceneContext->CreateChildEntity(entity, "Entity");
					opened = true;
				}

				ImGui::Separator();

				if (ImGui::MenuItem("Delete", "Delete")) {
					deleted = true;
				}

				if (ImGui::MenuItem("Rename")) {
					m_RenameContext = entity;
					Editor::Get().SetSelectionContext(entity);
				}

				ImGui::EndPopup();
			}

			ImGui::PopStyleVar();
		}

		// Drag & drop
		{
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
				ImGui::Text(name);
				ImGui::SetDragDropPayload("SceneEntity", &entity, sizeof(Entity));
				ImGui::EndDragDropSource();
			}

			if (ImGui::BeginDragDropTarget()) {
				const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneEntity", ImGuiDragDropFlags_AcceptNoDrawDefaultRect);

				if (payload) {
					const Entity& droppedEntity = *(Entity*)payload->Data;
					m_SceneContext->ParentEntity(droppedEntity, entity);
				}

				ImGui::EndDragDropTarget();
			}
		}

		// Draw child nodes
		if (opened) {
			for (const auto child : entity.Children()) {
				DrawEntityNode(m_SceneContext->GetEntityWithUUID(child), filter);
			}

			ImGui::TreePop();
		}

		// Defer deletion until the end node drawing.
		if (deleted) {
			if (entity == m_SelectionContext) {
				Editor::Get().SetSelectionContext({});
			}

			m_SceneContext->DeleteEntity(entity);
		}
	}

	bool SceneHierarchyPanel::TreeNode(Entity entity, const char* label, bool& hovered, bool& clicked, ImGuiID id, ImGuiTreeNodeFlags flags)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0);
		ImGui::TableNextRow(0, UI::Description.TreeNodeHeight);
		ImGui::TableSetColumnIndex(0);

		const ImVec2 rowAreaMin = ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), 0).Min;
		const ImVec2 rowAreaMax = { ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), ImGui::TableGetColumnCount() - 1).Max.x, rowAreaMin.y + UI::Description.TreeNodeHeight +5.0f };

		ImGuiContext& g = *GImGui;

		// check if there are any active popups
		if (g.OpenPopupStack.Size > 0) {
			// disable hover behaviour, when a popup is active
			hovered = false;
			clicked = false;
		}
		else {
			ImGui::PushClipRect(rowAreaMin, rowAreaMax, false);
			hovered = UI::ItemHoverable({ rowAreaMin, rowAreaMax }, id);
			clicked = hovered && ImGui::IsMouseClicked(0);
			ImGui::PopClipRect();
		}

		// Mouse over arrow
		ImGuiWindow* window = ImGui::GetCurrentWindow();
		ImGuiStyle& style = ImGui::GetStyle();
		const char* labelEnd = ImGui::FindRenderedTextEnd(label);

		const ImVec2 padding = ((flags & ImGuiTreeNodeFlags_FramePadding)) ? style.FramePadding : ImVec2(style.FramePadding.x, ImMin(window->DC.CurrLineTextBaseOffset, style.FramePadding.y));
		const float textOffsetX = g.FontSize + padding.x * 2;
		ImVec2 textPos(window->DC.CursorPos.x + textOffsetX, window->DC.CursorPos.y + UI::Description.TreeNodeTextOffsetY);
		const bool leaf = (flags & ImGuiTreeNodeFlags_Leaf) != 0;

		window->DC.CurrLineSize.y = UI::Description.TreeNodeHeight;

		if (clicked) {
			// Mouse is hovering the arrow on the X axis && the node has children
			if ((g.IO.MousePos.x >= (textPos.x - textOffsetX) - style.TouchExtraPadding.x && g.IO.MousePos.x < (textPos.x - textOffsetX) + (g.FontSize + padding.x * 2.0f) + style.TouchExtraPadding.x) && leaf == false) {
				clicked = false;
				ImGui::SetNextItemOpen(!ImGui::TreeNodeBehaviorIsOpen(id));
			}
			else {
				ImGui::SetActiveID(id, window);
			}
		}

		// Set tree node background color
		if (flags & ImGuiTreeNodeFlags_Selected) {
			ImGui::TableSetBgColor(3, (ImColor)style.Colors[ImGuiCol_FrameBgActive], 0);
			ImGui::TableSetBgColor(3, (ImColor)style.Colors[ImGuiCol_FrameBgActive], 1);
		}
		else if (hovered) {
			ImGui::TableSetBgColor(3, (ImColor)style.Colors[ImGuiCol_FrameBgHovered], 0);
			ImGui::TableSetBgColor(3, (ImColor)style.Colors[ImGuiCol_FrameBgHovered], 1);
		}

		if (window->SkipItems) {
			ImGui::PopStyleVar();
			return false;
		}

		const bool open = ImGui::TreeNodeBehaviorIsOpen(id, flags);
		if (open && !g.NavIdIsAlive && (flags & ImGuiTreeNodeFlags_NavLeftJumpsBackHere) && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			window->DC.TreeJumpToParentOnPopMask |= (1 << window->DC.TreeDepth);
		}

		if (flags & ImGuiTreeNodeFlags_AllowItemOverlap) {
			ImGui::SetItemAllowOverlap();
		}

		// Render
		{
			// Column 0 
			// Toggle arrow
			float heightWithPadding = (UI::Description.TreeNodeHeight + 2.0f);
			int indent = indent = (textPos.x - window->Pos.x - 7) / heightWithPadding;
			
			if (!leaf) {
				ImGui::RenderArrow(window->DrawList, ImVec2(textPos.x - textOffsetX + padding.x - 2, textPos.y + 2.0f + g.FontSize * 0.15f), ImGui::GetColorU32(style.Colors[ImGuiCol_Text]), open ? ImGuiDir_Down : ImGuiDir_Right, 0.6f);
				indent--;
			}

			for (size_t i = 2; i < indent + 1; i++)
			{
				float x = window->Pos.x + i * heightWithPadding - 7;
				window->DrawList->AddLine({ x, textPos.y - 2 }, { x, textPos.y + UI::Description.TreeNodeHeight + 3 }, ImGui::GetColorU32(style.Colors[ImGuiCol_Text]));
			}

			// Icon
			UI::ShiftCursor(UI::Description.TreeNodeHeight + 4.0f, 3.0f);
			UI::Image(leaf ? m_FileTexture : m_FolderTexture, ImVec2(UI::Description.TreeNodeHeight, UI::Description.TreeNodeHeight));
			UI::ShiftCursorY(2);
			textPos.x += UI::Description.TreeNodeHeight;
			textPos.y += 1.0f;
			if (g.LogEnabled) {
				ImGui::LogRenderedText(&textPos, ">");
			}

			// Rename input field
			if (m_RenameContext && entity == m_SelectionContext) {
				UI::ShiftCursor(33, -UI::Description.TreeNodeHeight - 5.0f);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x); // TODO: check what looks / works better
				ImGui::SetKeyboardFocusHere();

				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, { 4.0f, 2.0f });
				ImGui::InputTextWithHint("##rename", entity.GetComponent<TagComponent>().Tag.c_str(), s_RenameBuffer, ENTITY_NAME_MAX_LENGTH);
				ImGui::PopStyleVar();

				if (ImGui::IsItemDeactivatedAfterEdit())
				{
					RenameEntity();
				}
			}
			// Node label
			else {
				ImGui::RenderText(textPos, label, labelEnd, false);
				// ImGui::RenderText(textPos, std::to_string(indent).c_str());
			}

			// Column 1
			// Draw entity components icons here
			//ImGui::TableSetColumnIndex(1);
			//UI::ShiftCursor(4.0f, 2.0f);
			//ImVec2 cursorPos = ImGui::GetCursorPos();

			//cursorPos.x += m_PropertiesColumnWidth - m_IconSize - 17;
			//ImGui::SetCursorPos(cursorPos);

			//for (size_t i = 0; i < m_CurrentIconCount; i++)
			//{
			//	UI::Image(m_TestTextureAttribute, { (float)m_IconSize, (float)m_IconSize });
			//	cursorPos.x -= m_IconSize + m_IconSpacing;
			//	ImGui::SetCursorPos(cursorPos);
			//}
		}

		ImGui::PushClipRect(rowAreaMin, rowAreaMax, false);
		ImGui::ItemAdd(ImRect(rowAreaMin, rowAreaMax), id);
		ImGui::PopClipRect();

		if (open && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			ImGui::TreePushOverrideID(id);
		}

		IMGUI_TEST_ENGINE_ITEM_INFO(id, label, window->DC.ItemFlags | (leaf ? 0 : ImGuiItemStatusFlags_Openable) | (open ? ImGuiItemStatusFlags_Opened : 0));
		ImGui::PopStyleVar();

		if (hovered && ImGui::IsMouseReleased(0)) {
			ImGui::ClearActiveID();
		}

		return open;
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

	bool SceneHierarchyPanel::IsMatchingSearch(const std::string& item, const std::string_view searchQuery, const bool caseSensitive, const bool stripWhiteSpaces, const bool stripUnderscores)
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
		if (m_RenameContext) {
			// Don't rename the entity if the name buffer isn't empty
			if (strlen(s_RenameBuffer) != 0) {
				m_RenameContext.GetComponent<TagComponent>().Tag = s_RenameBuffer;
			}
		}

		ClearRenameBuffer();
		m_RenameContext = {};
	}

	void SceneHierarchyPanel::ClearRenameBuffer()
	{
		memset(s_RenameBuffer, 0, ENTITY_NAME_MAX_LENGTH);
	}

	bool SceneHierarchyPanel::OnKeyPressed(KeyPressedEvent& event)
	{
		if (Input::IsKeyPressed(KEY_ESCAPE)) {
			if (m_RenameContext) {
				ClearRenameBuffer();
				m_RenameContext = {};
			}

			return true; // Stop the event from bubbling further.
		}

		return false;
	}
}