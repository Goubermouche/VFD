#include "pch.h"
#include "UI.h"

#include <imgui.h>

namespace fe {
	UIDesc UI::s_Description;
	bool UI::s_ListColorCurrentIsDark;
	Ref<Texture> UI::Widget::s_SearchIcon;

	// ids
	int UI::s_UIContextID = 0;
	uint32_t UI::s_Counter = 0;
	char UI::s_IDBuffer[16] = "##";

	void UI::Init()
	{
		auto& assetManager = Editor::Get().GetAssetManager();
		Widget::s_SearchIcon = assetManager->Get<TextureAsset>("Resources/Images/Editor/search.png")->GetTexture();
	}

	void UI::ShiftCursor(float x, float y)
	{
		const ImVec2 cursor = ImGui::GetCursorPos();
		ImGui::SetCursorPos(ImVec2(cursor.x + x, cursor.y + y));
	}

	void UI::ShiftCursorX(float value)
	{
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() + value);
	}

	void UI::ShiftCursorY(float value)
	{
		ImGui::SetCursorPosY(ImGui::GetCursorPosY() + value);
	}

	const char* UI::GenerateID()
	{
		_itoa(s_Counter++, s_IDBuffer + 2, 16);
		return s_IDBuffer;
	}

	void UI::PushID()
	{
		ImGui::PushID(s_UIContextID++);
		s_Counter = 0;
	}

	void UI::PopID()
	{
		ImGui::PopID();
		s_UIContextID--;
	}

	bool UI::ItemHoverable(const ImRect& bb, ImGuiID id)
	{
		auto g = ImGui::GetCurrentContext();

		if (g->CurrentWindow != g->HoveredWindow) {
			return false;
		}

		if (ImGui::IsMouseHoveringRect(bb.Min, bb.Max)) {
			ImGui::SetHoveredID(id);
			return true;
		}

		return false;
	}

	inline ImRect UI::RectExpanded(const ImRect& rect, float x, float y)
	{
		ImRect result = rect;
		result.Min.x -= x;
		result.Min.y -= y;
		result.Max.x += x;
		result.Max.y += y;
		return result;
	}

	inline ImRect UI::GetItemRect()
	{
		return ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
	}

	void UI::Image(Ref<Texture> texture, const ImVec2& size)
	{
		ImGui::Image((void*)(intptr_t)texture->GetRendererID(), size);
	}

	void UI::Image(Ref<Texture> texture, const ImVec2& size, const ImVec4& tintColor)
	{
		ImGui::Image((void*)(intptr_t)texture->GetRendererID(), size, ImVec2{ 0, 0 }, ImVec2{ 1, 1 }, tintColor);
	}

	bool UI::TreeNode(const char* label, bool& isHovered, bool& isClicked, ImGuiID id, ImGuiTreeNodeFlags flags)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0);
		ImGui::TableNextRow(0, s_Description.RowHeight);
		ImGui::TableSetColumnIndex(0);

		const ImVec2 rowAreaMin = ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), 0).Min;
		const ImVec2 rowAreaMax = { ImGui::TableGetCellBgRect(ImGui::GetCurrentTable(), ImGui::TableGetColumnCount() - 1).Max.x, rowAreaMin.y + s_Description.RowHeight + 2 };

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
		window->DC.CurrLineSize.y = s_Description.RowHeight;
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
			ImGui::TableSetBgColor(3, s_Description.SelectionActive, 0);
			ImGui::TableSetBgColor(3, s_Description.SelectionActive, 1);
		}
		else if (isHovered) {
			ImGui::TableSetBgColor(3, s_Description.ListBackgroundHovered, 0);
			ImGui::TableSetBgColor(3, s_Description.ListBackgroundHovered, 1);
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
				ImGui::RenderArrow(window->DrawList, ImVec2(textPos.x - textOffsetX + padding.x, textPos.y + g.FontSize * 0.15f), s_Description.ListToggleColor, isOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.6f);
			}
			textPos.y -= 1.0f;
			if (g.LogEnabled) {
				ImGui::LogRenderedText(&textPos, ">");
			}

			// Node label
			ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)s_Description.ListTextColor);
			ImGui::RenderText(textPos, label, labelEnd, false);
			ImGui::PopStyleColor();

			// Column 1
			// Draw entity components icons here
			ImGui::TableSetColumnIndex(1);
			UI::ShiftCursor(4.0f, 2.0f);
			ImGui::Text("///");
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

	void UI::TreeBackground()
	{
		float rowHeight = s_Description.RowHeight + 2.0f;
		ImDrawList* drawList = ImGui::GetWindowDrawList();
		ImVec2 windowSize = ImGui::GetWindowSize();
		ImVec2 cursorPos = ImGui::GetWindowPos();

		windowSize.x += 20;
		cursorPos.y -= ImGui::GetScrollY();

		uint32_t clippedCount = ImGui::GetScrollY() / 20 ;
		uint32_t rowCount = windowSize.y / rowHeight + 2;

		cursorPos.y += clippedCount * rowHeight;
		s_ListColorCurrentIsDark = clippedCount % 2 == 0;

		for (uint32_t i = 0; i < rowCount; i++)
		{
			drawList->AddRectFilled(
				cursorPos, { cursorPos.x + windowSize.x, cursorPos.y + rowHeight },
				s_ListColorCurrentIsDark ? s_Description.ListBackgroundDark : s_Description.ListBackgroundLight);
			
			cursorPos.y += rowHeight;

			s_ListColorCurrentIsDark = !s_ListColorCurrentIsDark;
		}
	}

	void UI::ItemActivityOutline(float rounding, ImColor active, ImColor inactive, ImColor hovered)
	{
		auto* drawList = ImGui::GetWindowDrawList();
		const ImRect rect = RectExpanded(GetItemRect(), 1.0f, 1.0f);
		// hovered 
		if (ImGui::IsItemHovered() && !ImGui::IsItemActive())
		{
			drawList->AddRect(rect.Min, rect.Max,
				hovered, rounding, 0, 1.0f);
		}
		// active
		if (ImGui::IsItemActive())
		{
			drawList->AddRect(rect.Min, rect.Max,
				active, rounding, 0, 1.0f);
		}
		else if (!ImGui::IsItemHovered())
		{
			drawList->AddRect(rect.Min, rect.Max,
				inactive, rounding, 0, 1.0f);
		}
	}

	bool UI::Widget::SearchBar(std::string& searchString, const char* hint, bool* grabFocus)
	{
		UI::PushID();
		UI::ShiftCursorY(1.0f);

		const bool layoutSuspended = []
		{
			ImGuiWindow* window = ImGui::GetCurrentWindow();
			if (window->DC.CurrentLayout)
			{
				ImGui::SuspendLayout();
				return true;
			}
			return false;
		}();

		bool modified = false;
		bool searching = false;

		const float areaPosX = ImGui::GetCursorPosX();
		const float framePaddingY = ImGui::GetStyle().FramePadding.y;

		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(22.0f, framePaddingY));
		char searchBuffer[256]{};
		strcpy_s<256>(searchBuffer, searchString.c_str());
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImU32)s_Description.InputFieldBackground);

		if (ImGui::InputText(GenerateID(), searchBuffer, 256))
		{
			searchString = searchBuffer;
			modified = true;
		}
		else if (ImGui::IsItemDeactivatedAfterEdit())
		{
			searchString = searchBuffer;
			modified = true;
		}

		searching = searchBuffer[0] != 0;

		if (grabFocus && *grabFocus)
		{
			if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
				&& !ImGui::IsAnyItemActive()
				&& !ImGui::IsMouseClicked(0))
			{
				ImGui::SetKeyboardFocusHere(-1);
			}

			if (ImGui::IsItemFocused()) {
				*grabFocus = false;
			}
		}

		UI::ItemActivityOutline(2.0f, s_Description.InputOutline, s_Description.InputOutline, s_Description.InputOutline);
		ImGui::SetItemAllowOverlap();

		ImGui::SameLine(areaPosX + 5.0f);


		if (layoutSuspended) {
			ImGui::ResumeLayout();
		}

		ImGui::BeginHorizontal(GenerateID(), ImGui::GetItemRectSize());

		// Search icon
		{
			const float iconYOffset = framePaddingY - 2;
			UI::ShiftCursorY(iconYOffset);
			UI::Image(s_SearchIcon, ImVec2(s_SearchIcon->GetWidth(), s_SearchIcon->GetHeight()) , {1, 1, 1, 1});
			UI::ShiftCursorX(4);
			UI::ShiftCursorY(-iconYOffset);

			// Hint
			if (!searching)
			{
				UI::ShiftCursorY(-framePaddingY + 1.0f);
				ImGui::TextUnformatted(hint);
				UI::ShiftCursorY(-1.0f);
			}
		}

		ImGui::Spring();

		ImGui::PopStyleColor();
		ImGui::EndHorizontal();

		ImGui::PopStyleVar(3);

		UI::PopID();

		return modified;
	}
}