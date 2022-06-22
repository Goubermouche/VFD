#include "pch.h"
#include "SceneHierarchyPanel.h"

#include <imgui_internal.h>
#include "FluidEngine/Platform/ImGui/ImGuiUtilities.h"
#include "FluidEngine/Editor/Editor.h"

namespace fe {
	SceneHierarchyPanel::SceneHierarchyPanel()
	{
	}

	void SceneHierarchyPanel::OnUpdate()
	{
		PROFILE_SCOPE;

		const float iconSectionWidth = 80.0f;

		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 0.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 13.0f);
		ImVec2 availibleSpace = ImGui::GetContentRegionAvail();
		ImRect windowRect = { ImGui::GetWindowContentRegionMin(), ImGui::GetWindowContentRegionMax() };
		ImGuiTableFlags tableFlags = ImGuiTableFlags_NoPadInnerX | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
		if (ImGui::BeginTable("##SceneHierarchyTable", 2, tableFlags, availibleSpace)) {
			ImGui::TableSetupColumn("##0", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, availibleSpace.x - iconSectionWidth);
			ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, iconSectionWidth);

			// Draw entity tree nodes
			for (auto entity : m_SceneContext->m_Registry.view<IDComponent, RelationshipComponent>())
			{
				Entity e(entity, m_SceneContext.Raw());
				if (e.GetParentUUID() == 0) {
					DrawEntityNode(e);
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
						Editor::Get().LoadScene();
					}
					if (ImGui::MenuItem("Save Scene", "Ctrl + Save")) {
						Editor::Get().SaveScene();
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
				Entity& entity = *(Entity*)payload->Data;
				m_SceneContext->UnparentEntity(entity);
				ImGui::ClearActiveID();
			}
			ImGui::EndDragDropTarget();
		}
		ImGui::PopStyleVar(2);
	}

	void SceneHierarchyPanel::DrawEntityNode(Entity entity)
	{
		const char* name = entity.GetComponent<TagComponent>().Tag.c_str();
		bool isSelected = entity == m_SelectionContext;
		bool isDeleted = false;
		const std::string strID = std::string(name) + std::to_string((uint32_t)entity);
		ImGuiTreeNodeFlags flags = (isSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanFullWidth;
		if (entity.Children().empty()) {
			flags |= ImGuiTreeNodeFlags_Leaf;
		}

		bool hovered, clicked;
		// Draw tree node
		bool opened = DrawTreeNode(name, &hovered, &clicked, ImGui::GetID(strID.c_str()), flags);

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
					Entity& droppedEntity = *(Entity*)payload->Data;
					m_SceneContext->ParentEntity(droppedEntity, entity);
				}
				ImGui::EndDragDropTarget();
			}
		}

		// Draw child nodes
		if (opened)
		{
			for (auto child : entity.Children()) {
				DrawEntityNode(m_SceneContext->GetEntityWithUUID(child));
			}

			ImGui::TreePop();
		}

		// Defer deletion until the end node drawing.
		if (isDeleted) {
			if (entity == m_SelectionContext) {
				Editor::Get().SetSelectionContext({});
			}
			m_SceneContext->DestroyEntity(entity);
		}
	}

	bool SceneHierarchyPanel::DrawTreeNode(const char* label, bool* outHovered, bool* outClicked, ImGuiID id, ImGuiTreeNodeFlags flags)
	{
		const float rowHeight = 18.0f;

		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0);
		ImGui::TableNextRow(0, rowHeight);
		ImGui::TableSetColumnIndex(0);
		
		const ImVec2 rowAreaMin = ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), 0).Min;
		const ImVec2 rowAreaMax = { ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), ImGui::TableGetColumnCount() - 1).Max.x, rowAreaMin.y + rowHeight + 2 };
		
		ImGuiContext& g = *GImGui;

		// check if there are any active popups
		if (g.OpenPopupStack.Size > 0) {
			// disable hover behaviour, when a popup is active
			*outHovered = false;
			*outClicked = false;
		}
		else {
			ImGui::PushClipRect(rowAreaMin, rowAreaMax, false);
			*outHovered = UI::ItemHoverable({ rowAreaMin, rowAreaMax }, id);
			*outClicked = *outHovered && ImGui::IsMouseClicked(0);
			ImGui::PopClipRect();
		}
		
		// Prevents rendering of items that are outside the clip rect (scroll mode)
		// Causes scroll bugs wehn a tree is open 
	/*	if (ImGui::IsClippedEx(ImRect(rowAreaMin, rowAreaMax), id)) {
			ImGui::PopStyleVar();
			return false;
		}*/

		// Mouse over arrow
		const bool isWindowFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
		auto* window = ImGui::GetCurrentWindow();
		window->DC.CurrLineSize.y = rowHeight;
		auto& style = ImGui::GetStyle();
		const char* labelEnd = ImGui::FindRenderedTextEnd(label);
		const ImVec2 labelSize = ImGui::CalcTextSize(label, labelEnd, false);
		const ImVec2 padding = ((flags & ImGuiTreeNodeFlags_FramePadding)) ? style.FramePadding : ImVec2(style.FramePadding.x, ImMin(window->DC.CurrLineTextBaseOffset, style.FramePadding.y));
		const float textOffsetX = g.FontSize + padding.x * 2;
		ImVec2 textPos(window->DC.CursorPos.x + textOffsetX, window->DC.CursorPos.y + 3.0f);
		const bool isLeaf = (flags & ImGuiTreeNodeFlags_Leaf) != 0;
		bool activeIdWasSet = false;

		if (*outClicked) {
			// Mouse is hovering the arrow on the X axis && the node has children
			if ((g.IO.MousePos.x >= (textPos.x - textOffsetX) - style.TouchExtraPadding.x && g.IO.MousePos.x < (textPos.x - textOffsetX) + (g.FontSize + padding.x * 2.0f) + style.TouchExtraPadding.x) && isLeaf == false) {
				*outClicked = false;
				ImGui::SetNextItemOpen(!ImGui::TreeNodeBehaviorIsOpen(id));
			}
			else {
				activeIdWasSet = true;
				ImGui::SetActiveID(id, window);
			}
		}

		// Set tree node background color
		if (flags & ImGuiTreeNodeFlags_Selected) {
			ImGui::TableSetBgColor(3, ImColor(0, 0, 255, 255), 0);
			ImGui::TableSetBgColor(3, ImColor(0, 0, 255, 255), 1);
		}
		else if (*outHovered) {
			ImGui::TableSetBgColor(3, ImColor(255, 0, 0, 255), 0);
			ImGui::TableSetBgColor(3, ImColor(255, 0, 0, 255), 1);
		}

		if (window->SkipItems) {
			ImGui::PopStyleVar();
			return false;
		}

		bool isOpen = ImGui::TreeNodeBehaviorIsOpen(id, flags);
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
				ImGui::RenderArrow(window->DrawList, ImVec2(textPos.x - textOffsetX + padding.x, textPos.y + g.FontSize * 0.15f), ImColor(255, 255, 255, 255), isOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.7f);
			}
			textPos.y -= 1.0f;
			if (g.LogEnabled) {
				ImGui::LogRenderedText(&textPos, ">");
			}

			// Node label
			ImGui::RenderText(textPos, label, labelEnd, false);

			// Column 1
			// Draw entity components icons here
			ImGui::TableSetColumnIndex(1);
			UI::ShiftCursor(4.0f, 2.0f);
			ImGui::Text("Entity");
		}

		ImGui::PushClipRect(rowAreaMin, rowAreaMax, false);
		ImGui::ItemAdd(ImRect(rowAreaMin, rowAreaMax), id);
		ImGui::PopClipRect();

		if (isOpen && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			ImGui::TreePushOverrideID(id);
		}

		IMGUI_TEST_ENGINE_ITEM_INFO(id, label, window->DC.ItemFlags | (isLeaf ? 0 : ImGuiItemStatusFlags_Openable) | (isOpen ? ImGuiItemStatusFlags_Opened : 0));
		ImGui::PopStyleVar();

		if (*outHovered && ImGui::IsMouseReleased(0)) {
			ImGui::ClearActiveID();
		}

		return isOpen;
	}
}