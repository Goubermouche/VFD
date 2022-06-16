#include "pch.h"
#include "SceneHierarchyPanel.h"

#include <imgui_internal.h>

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

	void SceneHierarchyPanel::DrawEntityNode(Entity entity)
	{
		const char* name = "Unnamed Entity";
		if (entity.HasComponent<TagComponent>()) {
			name = entity.GetComponent<TagComponent>().Tag.c_str();
		}

		// id
		const std::string strID = std::string(name) + std::to_string((uint32_t)entity);

		// flags
		ImGuiTreeNodeFlags flags = (false ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanFullWidth;
		if (entity.Children().empty()) {
			flags |= ImGuiTreeNodeFlags_Leaf;
		}

		bool opened = DrawTreeNode(name, ImGui::GetID(strID.c_str()), flags);

		if (opened)
		{
			for (auto child : entity.Children()) {
				DrawEntityNode(m_SceneContext->GetEntityWithUUID(child));
			}

			ImGui::TreePop();
		}
	}

	// TODO: clean this up a bit.
	bool SceneHierarchyPanel::DrawTreeNode(const char* label, ImGuiID id, ImGuiTreeNodeFlags flags)
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

		bool hovered = ImGui::ItemHoverable(ImRect(rowAreaMin, rowAreaMax), id + 500);
		bool clicked = hovered && ImGui::IsMouseClicked(0);
		bool held = hovered && ImGui::IsMouseDown(0);

		ImGui::SetItemAllowOverlap();
		ImGui::PopClipRect();

		if (held) {
			ImGui::TableSetBgColor(3, ImColor(0, 0, 255, 255), 0);
			ImGui::TableSetBgColor(3, ImColor(0, 0, 255, 255), 1);
		}
		else if (hovered) {
			ImGui::TableSetBgColor(3, ImColor(255, 0, 0, 255), 0);
			ImGui::TableSetBgColor(3, ImColor(255, 0, 0, 255), 1);
		}

	    const bool isWindowFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

		auto* window = ImGui::GetCurrentWindow();
		window->DC.CurrLineSize.y = rowHeight;

		ImGuiContext& g = *GImGui;
		auto& style = ImGui::GetStyle();
		const char* labelEnd = ImGui::FindRenderedTextEnd(label);
		const ImVec2 labelSize = ImGui::CalcTextSize(label, labelEnd, false);
		const ImVec2 padding = ((flags & ImGuiTreeNodeFlags_FramePadding)) ? style.FramePadding : ImVec2(style.FramePadding.x, ImMin(window->DC.CurrLineTextBaseOffset, style.FramePadding.y));
		const float textOffsetX = g.FontSize + padding.x * 2;           // Collapser arrow width + Spacing
		const float textOffsetY = 3; // Latch before ItemSize changes it
		const float textWidth = g.FontSize + (labelSize.x > 0.0f ? labelSize.x + padding.x * 2 : 0.0f);  // Include collapser
		ImVec2 textPos(window->DC.CursorPos.x + textOffsetX, window->DC.CursorPos.y + textOffsetY);
		const float arrowHitX1 = (textPos.x - textOffsetX) - style.TouchExtraPadding.x;
		const float arrowHitX2 = (textPos.x - textOffsetX) + (g.FontSize + padding.x * 2.0f) + style.TouchExtraPadding.x;
		const bool isMouseOverArrow = (g.IO.MousePos.x >= arrowHitX1 && g.IO.MousePos.x < arrowHitX2);

		bool previousState = ImGui::TreeNodeBehaviorIsOpen(id);

		if (isMouseOverArrow && clicked) {
			ImGui::SetNextItemOpen(!previousState);
		}

		// tree node
		if (window->SkipItems) {
			ImGui::PopStyleVar();
			return false;
		}

		ImGuiLastItemData& lastItem = g.LastItemData;
		const ImGuiStyle& styleR = g.Style;

		// We vertically grow up to current line height up the typical widget height.
		const float frameHeight = window->DC.CurrLineSize.y;
		ImRect frameBB;
		frameBB.Min.x = (flags & ImGuiTreeNodeFlags_SpanFullWidth) ? window->WorkRect.Min.x : window->DC.CursorPos.x;
		frameBB.Min.y = window->DC.CursorPos.y;
		frameBB.Max.x = window->WorkRect.Max.x;
		frameBB.Max.y = window->DC.CursorPos.y + frameHeight;

		ImGui::ItemSize(ImVec2(textWidth, frameHeight), padding.y);

		// For regular tree nodes, we arbitrary allow to click past 2 worth of ItemSpacing
		ImRect interactBB = frameBB;
		if (flags & (ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth) == 0) {
			interactBB.Max.x = frameBB.Min.x + textWidth + styleR.ItemSpacing.x * 2.0f;
		}

		// Store a flag for the current depth to tell if we will allow closing this node when navigating one of its child.
		// For this purpose we essentially compare if g.NavIdIsAlive went from 0 to 1 between TreeNode() and TreePop().
		// This is currently only support 32 level deep and we are fine with (1 << Depth) overflowing into a zero.
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

		// We allow clicking on the arrow section with keyboard modifiers held, in order to easily
		// allow browsing a tree while preserving selection with code implementing multi-selection patterns.
		// When clicking on the rest of the tree node we always disallow keyboard modifiers.
		if (window != g.HoveredWindow || !isMouseOverArrow) {
			buttonFlags |= ImGuiButtonFlags_NoKeyModifiers;
		}

		// Open behaviors can be altered with the _OpenOnArrow and _OnOnDoubleClick flags.
		// Some alteration have subtle effects (e.g. toggle on MouseUp vs MouseDown events) due to requirements for multi-selection and drag and drop support.
		// - Single-click on label = Toggle on MouseUp (default, when _OpenOnArrow=0)
		// - Single-click on arrow = Toggle on MouseDown (when _OpenOnArrow=0)
		// - Single-click on arrow = Toggle on MouseDown (when _OpenOnArrow=1)
		// - Double-click on label = Toggle on MouseDoubleClick (when _OpenOnDoubleClick=1)
		// - Double-click on arrow = Toggle on MouseDoubleClick (when _OpenOnDoubleClick=1 and _OpenOnArrow=0)
		// It is rather standard that arrow click react on Down rather than Up.
		// We set ImGuiButtonFlags_PressedOnClickRelease on OpenOnDoubleClick because we want the item to be active on the initial MouseDown in order for drag and drop to work.
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

		bool hovered2;
		bool pressed = ImGui::ButtonBehavior(interactBB, id, &hovered2, &held, buttonFlags);
		bool toggled = false;

		if (!isLeaf)
		{
			if (pressed && g.DragDropHoldJustPressedId != id)
			{
				if ((flags & (ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick)) == 0 || (g.NavActivateId == id)) {
					toggled = true;
				}
				if (flags & ImGuiTreeNodeFlags_OpenOnArrow) {
					toggled |= isMouseOverArrow && !g.NavDisableMouseHover; // Lightweight equivalent of IsMouseHoveringRect() since ButtonBehavior() already did the job
				}
				if ((flags & ImGuiTreeNodeFlags_OpenOnDoubleClick) && g.IO.MouseDoubleClicked[0]) {
					toggled = true;
				}
			}
			else if (pressed && g.DragDropHoldJustPressedId == id)
			{
				IM_ASSERT(buttonFlags & ImGuiButtonFlags_PressedOnDragDropHold);
				if (!isOpen) { // When using Drag and Drop "hold to open" we keep the node highlighted after opening, but never close it again.
					toggled = true;
				}
			}

			if (g.NavId == id && g.NavMoveDir == ImGuiDir_Left && isOpen)
			{
				toggled = true;
				ImGui::NavMoveRequestCancel();
			}

			if (g.NavId == id && g.NavMoveDir == ImGuiDir_Right && !isOpen) // If there's something upcoming on the line we may want to give it the priority?
			{
				toggled = true;
				ImGui::NavMoveRequestCancel();
			}

			if (toggled)
			{
				isOpen = !isOpen;
				window->DC.StateStorage->SetInt(id, isOpen);
				lastItem.StatusFlags |= ImGuiItemStatusFlags_ToggledOpen;
			}
		}

		if (flags & ImGuiTreeNodeFlags_AllowItemOverlap) {
			ImGui::SetItemAllowOverlap();
		}

		if (selected != wasSelected) {
			lastItem.StatusFlags |= ImGuiItemStatusFlags_ToggledSelection;
		}

		// Render
		ImGuiNavHighlightFlags nav_highlight_flags = ImGuiNavHighlightFlags_TypeThin;
		{
			// Unframed typed for tree nodes
			if (hovered2 || selected)
			{
				//if (held && hovered) HZ_CORE_WARN("held && hovered");
				//if(hovered && !selected && !held && !pressed && !toggled) HZ_CORE_WARN("hovered && !selected && !held");
				//else if(!selected) HZ_CORE_WARN("ImGuiCol_Header");

				const ImU32 bg_col = ImGui::GetColorU32((held && hovered2) ? ImGuiCol_HeaderActive : (hovered && !selected && !held && !pressed && !toggled) ? ImGuiCol_HeaderHovered : ImGuiCol_Header);
				ImGui::RenderFrame(frameBB.Min, frameBB.Max, bg_col, false);
				ImGui::RenderNavHighlight(frameBB, id, nav_highlight_flags);
			}
			if (flags & ImGuiTreeNodeFlags_Bullet) {
				ImGui::RenderBullet(window->DrawList, ImVec2(textPos.x - textOffsetX * 0.5f, textPos.y + g.FontSize * 0.5f), ImColor(255, 0, 0, 255));
			}
			else if (!isLeaf) {
				ImGui::RenderArrow(window->DrawList, ImVec2(textPos.x - textOffsetX + padding.x, textPos.y + g.FontSize * 0.15f), ImColor(0, 255, 0, 255), isOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.70f);
			}

			textPos.y -= 1.0f;


			if (g.LogEnabled) {
				ImGui::LogRenderedText(&textPos, ">");
			}

			ImGui::RenderText(textPos, label, labelEnd, false);
		}

		auto fillRowWithColour = [](const ImColor& colour)
		{
			for (int column = 0; column < ImGui::TableGetColumnCount(); column++) {
				ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, colour, column);
			}
		};

		if (isOpen && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
			ImGui::TreePushOverrideID(id);
		}

		IMGUI_TEST_ENGINE_ITEM_INFO(id, label, window->DC.ItemFlags | (isLeaf ? 0 : ImGuiItemStatusFlags_Openable) | (isOpen ? ImGuiItemStatusFlags_Opened : 0));

		ImGui::PopStyleVar();

		return isOpen;
	}
}