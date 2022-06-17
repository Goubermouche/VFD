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

		if (ImGui::Begin(m_Name.c_str())) {
			m_Hovered = ImGui::IsWindowHovered();

			ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 0.0f));

			const float iconSectionWidth = 80;
			ImVec2 availibleSpace = ImGui::GetContentRegionAvail();

			ImGuiTableFlags tableFlags = ImGuiTableFlags_NoPadInnerX | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
			if (ImGui::BeginTable("##SceneHierarchyTable", 2, tableFlags, ImGui::GetContentRegionAvail())) {
				ImGui::TableSetupColumn("##0", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, availibleSpace.x - iconSectionWidth);
				ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, iconSectionWidth);

				for (auto entity : m_SceneContext->m_Registry.view<IDComponent, RelationshipComponent>())
				{
					Entity e(entity, m_SceneContext.Raw());
					if (e.GetParentUUID() == 0) {
						DrawEntityNode(e);
					}
				}

				ImGui::EndTable();
			}

			ImGui::PopStyleVar(1);
		}

		ImGui::End();
	}

	void SceneHierarchyPanel::OnEvent(Event& e)
	{

	}

	void SceneHierarchyPanel::SetSceneContext(Ref<Scene> context)
	{
		m_SceneContext = context;
	}

	void SceneHierarchyPanel::SetSelectionContext(Entity selectionContext)
	{
		m_SelectionContext = selectionContext;
	}

	void SceneHierarchyPanel::DrawEntityNode(Entity entity)
	{
		const char* name = "Unnamed Entity";
		if (entity.HasComponent<TagComponent>()) {
			name = entity.GetComponent<TagComponent>().Tag.c_str();
		}

		bool isSelected = entity == m_SelectionContext;

		// id
		const std::string strID = std::string(name) + std::to_string((uint32_t)entity);

		// flags
		ImGuiTreeNodeFlags flags = (isSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanFullWidth;
		if (entity.Children().empty()) {
			flags |= ImGuiTreeNodeFlags_Leaf;
		}


		bool hovered, clicked;
		bool opened = DrawTreeNode(name, &hovered, &clicked, ImGui::GetID(strID.c_str()), flags);

		if (clicked)
		{
			Editor::Get().OnSelectionContextChanged(entity);
		}

		if (opened)
		{
			for (auto child : entity.Children()) {
				DrawEntityNode(m_SceneContext->GetEntityWithUUID(child));
			}

			ImGui::TreePop();
		}
	}

	// TODO: clean this up a bit.
	// BUGS: while the scrollbar is active hover behaviour is wonky.
	bool SceneHierarchyPanel::DrawTreeNode(const char* label, bool* outHovered, bool* outClicked, ImGuiID id, ImGuiTreeNodeFlags flags)
	{
		const float rowHeight = 18.0f;

		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0);
		ImGui::TableNextRow(0, rowHeight);
		ImGui::TableSetColumnIndex(0);

		ImVec2 maxRegion = ImGui::GetContentRegionMax();

		const ImVec2 rowAreaMin = ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), 0).Min;
		const ImVec2 rowAreaMax = { ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), ImGui::TableGetColumnCount() - 1).Max.x + 60,
									rowAreaMin.y + rowHeight };

		ImGui::PushClipRect(rowAreaMin, rowAreaMax, false);

		*outHovered = ImGui::ItemHoverable(ImRect(rowAreaMin, rowAreaMax), id + 500);
		*outClicked = *outHovered && ImGui::IsMouseClicked(0);
		bool held = *outHovered && ImGui::IsMouseDown(0);

		ImGui::SetItemAllowOverlap();
		ImGui::PopClipRect();

		// selected
		if (flags & ImGuiTreeNodeFlags_Selected) {
			// Set the color for hovered & focused rows here
			/*if (*outHovered) {
				ImGui::TableSetBgColor(3, ImColor(0, 0, 128, 255), 0);
				ImGui::TableSetBgColor(3, ImColor(0, 0, 128, 255), 1);
			}*/
			/*else*/ {
				ImGui::TableSetBgColor(3, ImColor(0, 0, 255, 255), 0);
				ImGui::TableSetBgColor(3, ImColor(0, 0, 255, 255), 1);
			}
		}
		else {
			if (*outHovered) {
				ImGui::TableSetBgColor(3, ImColor(255, 0, 0, 255), 0);
				ImGui::TableSetBgColor(3, ImColor(255, 0, 0, 255), 1);
			}
		}
		
	    const bool isWindowFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

		auto* window = ImGui::GetCurrentWindow();
		window->DC.CurrLineSize.y = rowHeight;

		ImGuiContext& g = *GImGui;
		auto& style = ImGui::GetStyle();
		const char* labelEnd = ImGui::FindRenderedTextEnd(label);
		const ImVec2 labelSize = ImGui::CalcTextSize(label, labelEnd, false);
		const ImVec2 padding = ((flags & ImGuiTreeNodeFlags_FramePadding)) ? style.FramePadding : ImVec2(style.FramePadding.x, ImMin(window->DC.CurrLineTextBaseOffset, style.FramePadding.y));
		const float textOffsetX = g.FontSize + padding.x * 2;
		const float textOffsetY = 3;
		const float textWidth = g.FontSize + (labelSize.x > 0.0f ? labelSize.x + padding.x * 2 : 0.0f);
		ImVec2 textPos(window->DC.CursorPos.x + textOffsetX, window->DC.CursorPos.y + textOffsetY);
		const float arrowHitX1 = (textPos.x - textOffsetX) - style.TouchExtraPadding.x;
		const float arrowHitX2 = (textPos.x - textOffsetX) + (g.FontSize + padding.x * 2.0f) + style.TouchExtraPadding.x;
		const bool isMouseOverArrow = (g.IO.MousePos.x >= arrowHitX1 && g.IO.MousePos.x < arrowHitX2);

		bool previousState = ImGui::TreeNodeBehaviorIsOpen(id);

		if (isMouseOverArrow && *outClicked) {
			ImGui::SetNextItemOpen(!previousState);
		}

		// tree node
		if (window->SkipItems) {
			ImGui::PopStyleVar();
			return false;
		}

		ImGuiLastItemData& lastItem = g.LastItemData;
		const ImGuiStyle& styleR = g.Style;

		const float frameHeight = window->DC.CurrLineSize.y;
		ImRect frameBB;
		frameBB.Min.x = (flags & ImGuiTreeNodeFlags_SpanFullWidth) ? window->WorkRect.Min.x : window->DC.CursorPos.x;
		frameBB.Min.y = window->DC.CursorPos.y;
		frameBB.Max.x = window->WorkRect.Max.x;
		frameBB.Max.y = window->DC.CursorPos.y + frameHeight;

		ImGui::ItemSize(ImVec2(textWidth, frameHeight), padding.y);

		ImRect interactBB = frameBB;
		if (flags & (ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth) == 0) {
			interactBB.Max.x = frameBB.Min.x + textWidth + styleR.ItemSpacing.x * 2.0f;
		}

		const bool isLeaf = (flags & ImGuiTreeNodeFlags_Leaf) != 0;
		bool isOpen = ImGui::TreeNodeBehaviorIsOpen(id, flags);
		if (isOpen && !g.NavIdIsAlive && (flags & ImGuiTreeNodeFlags_NavLeftJumpsBackHere) && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			window->DC.TreeJumpToParentOnPopMask |= (1 << window->DC.TreeDepth);
		}

		bool itemAdd = ImGui::ItemAdd(interactBB, id);
		lastItem.StatusFlags |= ImGuiItemStatusFlags_HasDisplayRect;
		lastItem.DisplayRect = frameBB;

		if (!itemAdd)
		{
			if (isOpen && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
				ImGui::TreePushOverrideID(id);
			}

			ImGui::PopStyleVar();

			IMGUI_TEST_ENGINE_ITEM_INFO(lastItem.ID, label, lastItem.StatusFlags | (isLeaf ? 0 : ImGuiItemStatusFlags_Openable) | (isOpen ? ImGuiItemStatusFlags_Opened : 0));
			return isOpen;
		}

		ImGuiButtonFlags buttonFlags = ImGuiTreeNodeFlags_None;
		if (flags & ImGuiTreeNodeFlags_AllowItemOverlap) {
			buttonFlags |= ImGuiButtonFlags_AllowItemOverlap;
		}
		if (!isLeaf) {
			buttonFlags |= ImGuiButtonFlags_PressedOnDragDropHold;
		}
		
		if (window != g.HoveredWindow || !isMouseOverArrow) {
			buttonFlags |= ImGuiButtonFlags_NoKeyModifiers;
		}

		if (isMouseOverArrow) {
			buttonFlags |= ImGuiButtonFlags_PressedOnClick;
		}
		else if (flags & ImGuiTreeNodeFlags_OpenOnDoubleClick) {
			buttonFlags |= ImGuiButtonFlags_PressedOnClickRelease | ImGuiButtonFlags_PressedOnDoubleClick;
		}
		else {
			buttonFlags |= ImGuiButtonFlags_PressedOnClickRelease;
		}

		bool selected = (flags & ImGuiTreeNodeFlags_Selected) != 0;
		const bool wasSelected = selected;

		if (flags & ImGuiTreeNodeFlags_AllowItemOverlap) {
			ImGui::SetItemAllowOverlap();
		}

		if (selected != wasSelected) {
			lastItem.StatusFlags |= ImGuiItemStatusFlags_ToggledSelection;
		}

		// Render
		{
			// Column 0
			if (flags & ImGuiTreeNodeFlags_Bullet) {
				// Unused
				ImGui::RenderBullet(window->DrawList, ImVec2(textPos.x - textOffsetX * 0.5f, textPos.y + g.FontSize * 0.5f), ImColor(255, 0, 0, 255));
			}
			else if (!isLeaf) {
				ImGui::RenderArrow(window->DrawList, ImVec2(textPos.x - textOffsetX + padding.x, textPos.y + g.FontSize * 0.15f), ImColor(255, 255, 255, 255), isOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.70f);
			}

			textPos.y -= 1.0f;

			if (g.LogEnabled) {
				ImGui::LogRenderedText(&textPos, ">");
			}

			ImGui::RenderText(textPos, label, labelEnd, false);

			ImGui::TableSetColumnIndex(1);

			// Column 1
			// Draw entity components icons
			UI::ShiftCursor(4, 2);
			ImGui::Text("Entity");
		}

		if (isOpen && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			ImGui::TreePushOverrideID(id);
		}

		IMGUI_TEST_ENGINE_ITEM_INFO(id, label, window->DC.ItemFlags | (isLeaf ? 0 : ImGuiItemStatusFlags_Openable) | (isOpen ? ImGuiItemStatusFlags_Opened : 0));
		ImGui::PopStyleVar();

		return isOpen;
	}
}