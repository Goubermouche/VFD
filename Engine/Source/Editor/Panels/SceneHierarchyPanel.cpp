#include "pch.h"
#include "SceneHierarchyPanel.h"

#include <imgui_internal.h>
#include "UI/UI.h"
#include "Editor/Editor.h"
#include "Utility/String.h"

namespace fe {
	void SceneHierarchyPanel::OnUpdate()
	{
		const float iconSectionWidth = 80.0f;

		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 0.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 10.0f);
		ImGui::PushStyleColor(ImGuiCol_TableBorderLight, { 0, 0, 0, 0 });
		ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, { 0.314,0.314,0.314, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, { 0.37,0.37,0.37, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, { 0.188,0.188,0.188, 1.0f });
		ImVec2 availableSpace = ImGui::GetContentRegionAvail();

		ImGui::SetNextItemWidth(availableSpace.x - 10.0f);
		UI::ShiftCursor(5, 3);
		std::string entitySearchString;
		UI::Widget::SearchBar(entitySearchString, "Filter");
		UI::ShiftCursor(-5, 4);

		availableSpace = ImGui::GetContentRegionAvail();
		const ImRect windowRect = { ImGui::GetWindowContentRegionMin(), ImGui::GetWindowContentRegionMax() };
		constexpr ImGuiTableFlags tableFlags = ImGuiTableFlags_NoPadInnerX | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;

		if (ImGui::BeginTable("##SceneHierarchyTable", 2, tableFlags, availableSpace)) {
			ImGui::TableSetupColumn("##0", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, availableSpace.x - iconSectionWidth);
			ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, iconSectionWidth);
			UI::TreeBackground();

			// Draw entity tree nodes
			for (auto entity : m_SceneContext->m_Registry.view<IDComponent, RelationshipComponent>())
			{
				Entity e(entity, m_SceneContext.Raw());
				if (e.GetParentUUID() == 0) {
					DrawEntityNode(e, entitySearchString);
				}
			}

			// Context menu
			{
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 2.0f, 2.0f });
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 4.0f });
				if (ImGui::BeginPopupContextWindow(nullptr, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
					if (ImGui::MenuItem("Create Empty")) {
						m_SceneContext->CreateEntity("Entity");
					}

					ImGui::Separator();
					if (ImGui::MenuItem("Open Scene")) {
						Editor::Get().LoadSceneContext();
					}

					if (ImGui::MenuItem("Save Scene", "Ctrl + Save")) {
						Editor::Get().SaveCurrentSceneContext();
					}

					ImGui::EndPopup();
				}

				ImGui::PopStyleVar(2);
			}
			

			ImGui::EndTable();
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

		ImGui::PopStyleVar(2);
		ImGui::PopStyleColor(4);
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
		const std::string strID = std::string(name) + std::to_string((uint32_t)entity);
		ImGuiTreeNodeFlags flags = (isSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth;

		if (entity.Children().empty()) {
			flags |= ImGuiTreeNodeFlags_Leaf;
		}

		bool hovered;
		bool clicked;

		// Draw tree node
		const bool opened = UI::TreeNode(name, hovered, clicked, ImGui::GetID(strID.c_str()), flags);

		if (clicked)
		{
			Editor::Get().SetSelectionContext(entity);
		}

		// Context menu
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 2.0f, 2.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 4.0f });
			if (ImGui::BeginPopupContextItem()) {
				if (ImGui::MenuItem("Create Empty")) {
					m_SceneContext->CreateChildEntity(entity, "Entity");
				}
				if (ImGui::MenuItem("Delete", "Delete")) {
					isDeleted = true;
				}
				ImGui::EndPopup();
			}
			ImGui::PopStyleVar(2);
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
}