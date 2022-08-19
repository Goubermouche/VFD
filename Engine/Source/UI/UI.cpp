#include "pch.h"
#include "UI.h"

#include <imgui.h>
#include "UI/ImGui/ImGuiRenderer.h" 
#include "UI/ImGui/ImGuiGLFWBackend.h"
#include <imgui_internal.h>
#include "Core/Application.h"

namespace fe {
	UIDesc UI::Description;
	bool UI::s_ListColorCurrentIsDark;
	Ref<Texture> UI::Widget::s_SearchIcon;
	Ref<Texture> UI::Widget::s_CloseIcon;

	// ids
	int UI::s_UIContextID = 0;
	uint32_t UI::s_Counter = 0;
	char UI::s_IDBuffer[16] = "##";

	void UI::Init()
	{
		auto& assetManager = Editor::Get().GetAssetManager();
		Widget::s_SearchIcon = assetManager->Get<TextureAsset>("Resources/Images/Editor/search.png")->GetTexture();
		Widget::s_CloseIcon = assetManager->Get<TextureAsset>("Resources/Images/Editor/close.png")->GetTexture();

		// Initialize the ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow()), true);
		ImGui_ImplOpenGL3_Init("#version 410"); // Use GLSL version 410

		// IO
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigWindowsMoveFromTitleBarOnly = true;

		// Font
		ImFontConfig fontConfig;
		fontConfig.OversampleH = 5;
		fontConfig.OversampleV = 5;
		static const ImWchar ranges[] =
		{
			0x0020, 0x00FF, // Basic Latin + Latin Supplement
			0x2200, 0x22FF, // Mathematical Operators
			0x0370, 0x03FF, // Greek and Coptic
			0,
		};

		io.FontDefault = io.Fonts->AddFontFromFileTTF("Resources/Fonts/OpenSans/OpenSans-SemiBold.ttf", 15.0f, &fontConfig, ranges);

		// Style
		ImGui::StyleColorsDark();
		ImGuiStyle& style = ImGui::GetStyle();
		style.ItemSpacing = { 0.0f, 0.0f };
		style.WindowPadding = { 0.0f, 0.0f };
		style.ScrollbarRounding = 2.0f;
		style.FrameBorderSize = 1.0f;
		style.TabRounding = 0.0f;
		style.WindowMenuButtonPosition = ImGuiDir_None;
		style.WindowRounding = 0.0f;
		//style.WindowMinSize = { 100.0f, 109.0f };
		style.WindowBorderSize = 0;
		style.ChildBorderSize = 0;
		style.FrameBorderSize = 0;

		style.IndentSpacing = Description.TreeNodeHeight + 2.0f;

		// style.Colors[ImGuiCol_WindowBg] = Description.WindowBackground;
		style.Colors[ImGuiCol_TitleBgCollapsed] = Description.WindowTitleBackground;
		style.Colors[ImGuiCol_TitleBg] = Description.WindowTitleBackground;
		style.Colors[ImGuiCol_TitleBgActive] = Description.WindowTitleBackgroundFocused;
		
		// style.Colors[ImGuiCol_Tab] = Description.TabBackground;
		// style.Colors[ImGuiCol_TabUnfocused] = Description.TabBackground;
		// style.Colors[ImGuiCol_TabUnfocusedActive] = Description.TabBackground;
		// style.Colors[ImGuiCol_TabHovered] = Description.TabBackgroundHovered;
		// style.Colors[ImGuiCol_TabActive] = Description.TabBackgroundFocused;
		   
		// style.Colors[ImGuiCol_Separator] = Description.Transparent;
		// style.Colors[ImGuiCol_SeparatorActive] = Description.Separator
		// style.Colors[ImGuiCol_SeparatorHovered] = Description.Separator;

		// style.Colors[ImGuiCol_NavHighlight] = Description.Transparent;

		LOG("ImGui initialized successfully", "editor][ImGui");
	}

	void UI::ShiftCursor(const float x, const float y)
	{
		const ImVec2 cursor = ImGui::GetCursorPos();
		ImGui::SetCursorPos(ImVec2(cursor.x + x, cursor.y + y));
	}

	void UI::ShiftCursorX(const float value)
	{
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() + value);
	}

	void UI::ShiftCursorY(const float value)
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

	bool UI::ItemHoverable(const ImRect& bb, const ImGuiID id)
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

	inline ImRect UI::RectExpanded(const ImRect& rect,const float x, const float y)
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

	bool UI::IsRootOfOpenMenuSet()
	{
		ImGuiContext& g = *GImGui;
		if ((g.OpenPopupStack.Size <= g.BeginPopupStack.Size) || (g.CurrentWindow->Flags & ImGuiWindowFlags_ChildMenu)) {
			return false;
		}

		const ImGuiPopupData* upper_popup = &g.OpenPopupStack[g.BeginPopupStack.Size];
		return upper_popup->Window && (upper_popup->Window->Flags & ImGuiWindowFlags_ChildMenu);
	}

	void UI::Image(Ref<Texture> texture, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1)
	{
		ImGui::Image((void*)(intptr_t)texture->GetRendererID(), size, uv0, uv1);
	}

	void UI::Image(Ref<Texture> texture, const ImVec2& size, const ImVec4& tintColor)
	{
		ImGui::Image((void*)(intptr_t)texture->GetRendererID(), size, ImVec2{ 0, 0 }, ImVec2{ 1, 1 }, tintColor);
	}

	void UI::DrawButtonImage(Ref<Texture> imageNormal, Ref<Texture> imageHovered, Ref<Texture> imagePressed, ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed, ImVec2 rectMin, ImVec2 rectMax)
	{
		auto* drawList = ImGui::GetWindowDrawList();
		if (ImGui::IsItemActive()) {
			drawList->AddImage((void*)(intptr_t)imagePressed->GetRendererID(), rectMin, rectMax, ImVec2(0, 0), ImVec2(1, 1), tintPressed);
		}
		else if (ImGui::IsItemHovered()) {
			drawList->AddImage((void*)(intptr_t)imageHovered->GetRendererID(), rectMin, rectMax, ImVec2(0, 0), ImVec2(1, 1), tintHovered);
		}
		else {
			drawList->AddImage((void*)(intptr_t)imageNormal->GetRendererID(), rectMin, rectMax, ImVec2(0, 0), ImVec2(1, 1), tintNormal);
		}
	}

	void UI::DrawButtonImage(Ref<Texture> texture, ImColor tintNormal, ImColor tintHovered, ImColor tintPressed, ImRect rectangle)
	{
		DrawButtonImage(texture, texture, texture, tintNormal, tintHovered, tintPressed, rectangle.Min, rectangle.Max);
	}

	void UI::DrawButtonImage(Ref<Texture> imageNormal, Ref<Texture> imageHovered, ImColor tintNormal, ImColor tintHovered, ImColor tintPressed, ImRect rectangle)
	{
		DrawButtonImage(imageNormal, imageHovered, imageHovered, tintNormal, tintHovered, tintPressed, rectangle.Min, rectangle.Max);
	}

	void UI::Separator()
	{
		ShiftCursorY(2);
		ImGui::Separator();
		ShiftCursorY(2);
	}

	bool UI::BeginMenu(const char* label,const bool enabled)
	{
		//ImGuiWindow* window = ImGui::GetCurrentWindow();
		//if (window->SkipItems) {
		//	return false;
		//}

		//ImGuiContext& g = *GImGui;
		//const ImGuiStyle& style = g.Style;
		//const ImGuiID id = window->GetID(label);
		//bool childMenuIsOpen = ImGui::IsPopupOpen(id, ImGuiPopupFlags_None);

		//ImGuiWindowFlags flags = ImGuiWindowFlags_ChildMenu | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNavFocus;
		//if (window->Flags & ImGuiWindowFlags_ChildMenu)
		//{
		//	flags |= ImGuiWindowFlags_ChildWindow;
		//}

		//if (g.MenusIdSubmittedThisFrame.contains(id))
		//{
		//	if (childMenuIsOpen)
		//	{
		//		childMenuIsOpen = ImGui::BeginPopupEx(id, flags); 
		//	}
		//	else
		//	{
		//		g.NextWindowData.ClearFlags(); 
		//	}
		//	return childMenuIsOpen;
		//}

		//g.MenusIdSubmittedThisFrame.push_back(id);
		//const ImVec2 labelSize = ImGui::CalcTextSize(label, nullptr, true);

		//const bool menuSetOpen = IsRootOfOpenMenuSet();
		//ImGuiWindow* backedNavWindow = g.NavWindow;
		//if (menuSetOpen)
		//{
		//	g.NavWindow = window;
		//}

		//ImVec2 popupPos;
		//const ImVec2 cursorPos = window->DC.CursorPos;

		//ImGui::PushID(label);
		//if (enabled == false)
		//{
		//	ImGui::BeginDisabled();
		//}

		//const ImGuiMenuColumns* offsets = &window->DC.MenuColumns;
		//constexpr ImGuiSelectableFlags selectableFlags = ImGuiSelectableFlags_NoHoldingActiveID | ImGuiSelectableFlags_SelectOnClick | ImGuiSelectableFlags_DontClosePopups;
		//bool isHovered = false;
		//bool isPressed = false;

		//if (window->DC.LayoutType == ImGuiLayoutType_Horizontal)
		//{
		//	ASSERT("not implemented");
		//	// Menu inside an horizontal menu bar
		//	// Selectable extend their highlight by half ItemSpacing in each direction.
		//	// For ChildMenu, the popup position will be overwritten by the call to FindBestWindowPosForPopup() in Begin()
		//	//popup_pos = ImVec2(pos.x - 1.0f - IM_FLOOR(style.ItemSpacing.x * 0.5f), pos.y - style.FramePadding.y + window->MenuBarHeight());
		//	//window->DC.CursorPos.x += IM_FLOOR(style.ItemSpacing.x * 0.5f);
		//	//ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(style.ItemSpacing.x * 2.0f, style.ItemSpacing.y));
		//	//float w = label_size.x;
		//	//ImVec2 text_pos(window->DC.CursorPos.x + offsets->OffsetLabel, window->DC.CursorPos.y + window->DC.CurrLineTextBaseOffset);
		//	//pressed = ImGui::Selectable("", menu_is_open, selectable_flags, ImVec2(w, 0.0f));
		//	//ImGui::RenderText(text_pos, label);
		//	//ImGui::PopStyleVar();
		//	//window->DC.CursorPos.x += IM_FLOOR(style.ItemSpacing.x * (-1.0f + 0.5f)); // -1 spacing to compensate the spacing added when Selectable() did a SameLine(). It would also work to call SameLine() ourselves after the PopStyleVar().
		//}
		//else
		//{
		//	popupPos = ImVec2(cursorPos.x, cursorPos.y - style.WindowPadding.y);
		//	const float checkMarkWidth = IM_FLOOR(g.FontSize * 1.20f);

		//	const float minWidth = window->DC.MenuColumns.DeclColumns(Description.ContextMenuIconIndent, Description.ContextMenuLabelWidth, Description.ContextMenuShortcutWidth, 0);
		//	const float stretchWidth = ImMax(0.0f, ImGui::GetContentRegionAvail().x - minWidth);

		//	ImVec2 textPos(window->DC.CursorPos.x + offsets->OffsetLabel, window->DC.CursorPos.y + window->DC.CurrLineTextBaseOffset);
		//	isPressed = ImGui::Selectable("", childMenuIsOpen, selectableFlags | ImGuiSelectableFlags_SpanAvailWidth, ImVec2(minWidth, 0.0f));
		//	isHovered = ImGui::IsItemHovered();
		//	const ImU32 color = isHovered || childMenuIsOpen ? ImU32(Description.ContextMenuButtonBackgroundHovered) : ImU32(Description.ContextMenuButtonBackground);
		//	ImGui::RenderFrame(ImVec2(cursorPos.x - 4, cursorPos.y - 2), ImVec2(cursorPos.x + minWidth + 4, cursorPos.y + labelSize.y + 2), color, false, 2);

		//	ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Description.ContextMenuLabel);
		//	ImGui::RenderText({ textPos.x, textPos.y}, label);
		//	ImGui::PopStyleColor();

		//	ImGui::RenderArrow(window->DrawList, ImVec2(offsets->OffsetShortcut + stretchWidth + cursorPos.x + Description.ContextMenuShortcutWidth - checkMarkWidth + 11, cursorPos.y + 3), (ImU32)(Description.ContextMenuArrow), ImGuiDir_Right, 0.6f);
		//}

		//if (!enabled)
		//{
		//	ImGui::EndDisabled();
		//}

		//if (menuSetOpen)
		//{
		//	g.NavWindow = backedNavWindow;
		//}

		//bool wantsToOpen = false;
		//bool wantsToClose = false;

		//if (window->DC.LayoutType == ImGuiLayoutType_Vertical)
		//{
		//	bool movingTowardsChildMenu = false;
		//	const ImGuiWindow* childMenuWindows = (g.BeginPopupStack.Size < g.OpenPopupStack.Size&& g.OpenPopupStack[g.BeginPopupStack.Size].SourceWindow == window) ? g.OpenPopupStack[g.BeginPopupStack.Size].Window : nullptr;
		//	if (g.HoveredWindow == window && childMenuWindows != nullptr && !(window->Flags & ImGuiWindowFlags_MenuBar))
		//	{
		//		const float refUnit = g.FontSize; // FIXME-DPI
		//		const ImRect nextWindowRect = childMenuWindows->Rect();
		//		const ImVec2 v1 = g.IO.MousePos;
		//		const ImVec2 v2 = g.IO.MouseDelta;
		//		ImVec2 ta = {v1.x - v2.x, v1.y - v2.y};
		//		ImVec2 tb = (window->Pos.x < childMenuWindows->Pos.x) ? nextWindowRect.GetTL() : nextWindowRect.GetTR();
		//		ImVec2 tc = (window->Pos.x < childMenuWindows->Pos.x) ? nextWindowRect.GetBL() : nextWindowRect.GetBR();
		//		const float extra = ImClamp(ImFabs(ta.x - tb.x) * 0.30f, refUnit * 0.5f, refUnit * 2.5f);
		//		ta.x += (window->Pos.x < childMenuWindows->Pos.x) ? -0.5f : +0.5f;
		//		tb.y = ta.y + ImMax((tb.y - extra) - ta.y, -refUnit * 8.0f);
		//		tc.y = ta.y + ImMin((tc.y + extra) - ta.y, +refUnit * 8.0f);
		//		movingTowardsChildMenu = ImTriangleContainsPoint(ta, tb, tc, g.IO.MousePos);
		//	}
		//	if (childMenuIsOpen && !isHovered && g.HoveredWindow == window && g.HoveredIdPreviousFrame != 0 && g.HoveredIdPreviousFrame != id && !movingTowardsChildMenu)
		//	{
		//		wantsToClose = true;
		//	}

		//	// Open
		//	if (!childMenuIsOpen && isPressed)
		//	{
		//		wantsToOpen = true;
		//	}
		//	else if (!childMenuIsOpen && isHovered && !movingTowardsChildMenu)
		//	{
		//		wantsToOpen = true;
		//	}
		//	if (g.NavId == id && g.NavMoveDir == ImGuiDir_Right)
		//	{
		//		wantsToOpen = true;
		//		ImGui::NavMoveRequestCancel();
		//	}
		//}
		//else
		//{
		//	ASSERT("not implemented")
		//	// Menu bar
		//	//if (menu_is_open && pressed && menuset_is_open) // Click an open menu again to close it
		//	//{
		//	//	want_close = true;
		//	//	want_open = menu_is_open = false;
		//	//}
		//	//else if (pressed || (hovered && menuset_is_open && !menu_is_open)) // First click to open, then hover to open others
		//	//{
		//	//	want_open = true;
		//	//}
		//	//else if (g.NavId == id && g.NavMoveDir == ImGuiDir_Down) // Nav-Down to open
		//	//{
		//	//	want_open = true;
		//	//	ImGui::NavMoveRequestCancel();
		//	//}
		//}

		//if (!enabled) {
		//	wantsToClose = true;
		//}
		//if (wantsToClose && ImGui::IsPopupOpen(id, ImGuiPopupFlags_None)) {
		//	ImGui::ClosePopupToLevel(g.BeginPopupStack.Size, true);
		//}

		//IMGUI_TEST_ENGINE_ITEM_INFO(id, label, g.LastItemData.StatusFlags | ImGuiItemStatusFlags_Openable | (menu_is_open ? ImGuiItemStatusFlags_Opened : 0));

		//PopID(); 

		//if (!childMenuIsOpen && wantsToOpen && g.OpenPopupStack.Size > g.BeginPopupStack.Size)
		//{
		//	ImGui::OpenPopup(label);
		//	return false;
		//}

		//childMenuIsOpen |= wantsToOpen;
		//if (wantsToOpen) {
		//	ImGui::OpenPopup(label);
		//}

		//if (childMenuIsOpen)
		//{
		//	ImGui::SetNextWindowPos(popupPos, ImGuiCond_Always); // Note: this is super misleading! The value will serve as reference for FindBestWindowPosForPopup(), not actual pos.
		//	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, style.PopupRounding); // First level will use _PopupRounding, subsequent will use _ChildRounding
		//	childMenuIsOpen = ImGui::BeginPopupEx(id, flags); // menu_is_open can be 'false' when the popup is completely clipped (e.g. zero size display)
		//	ImGui::PopStyleVar();
		//}
		//else
		//{
		//	g.NextWindowData.ClearFlags(); // We behave like Begin() and need to consume those values
		//}

		//return childMenuIsOpen;
		return false;
	}

	bool UI::MenuItem(const char* label, const char* shortcut, const bool selected, const bool enabled)
	{
		//ImGuiWindow* window = ImGui::GetCurrentWindow();
		//if (window->SkipItems) {
		//	return false;
		//}

		//ImGuiContext& g = *GImGui;
		//const ImGuiStyle& style = g.Style;
		//const ImVec2 pos = window->DC.CursorPos;
		//const ImVec2 labelSize = ImGui::CalcTextSize(label, nullptr, true);

		//const bool menuSetOpen = IsRootOfOpenMenuSet();
		//bool isPressed = false;

		//ImGuiWindow* backed_nav_window = g.NavWindow;
		//if (menuSetOpen) {
		//	g.NavWindow = window;
		//}

		//ImGui::PushID(label);
		//if (!enabled) {
		//	ImGui::BeginDisabled();
		//}

		//constexpr  ImGuiSelectableFlags selectableFlags = ImGuiSelectableFlags_SelectOnRelease | ImGuiSelectableFlags_SetNavIdOnHover;
		//const ImGuiMenuColumns* offsets = &window->DC.MenuColumns;

		//const float shortcutWidth = (shortcut && shortcut[0]) ? ImGui::CalcTextSize(shortcut, NULL).x : 0.0f;
		//const float minWidth = window->DC.MenuColumns.DeclColumns(Description.ContextMenuIconIndent, Description.ContextMenuLabelWidth, Description.ContextMenuShortcutWidth, 1);
		//const float stretchWidth = ImMax(0.0f, ImGui::GetContentRegionAvail().x - minWidth);

		//isPressed = ImGui::Selectable("", false, selectableFlags | ImGuiSelectableFlags_SpanAvailWidth, ImVec2(minWidth, 0.0f));

		//const bool isHovered = ImGui::IsItemHovered();
		//const ImU32 color = isHovered ? ImU32(Description.ContextMenuButtonBackgroundHovered) : ImU32(Description.ContextMenuButtonBackground);
		//ImGui::RenderFrame(ImVec2(pos.x - 4, pos.y - 2), ImVec2(pos.x + minWidth + 4, pos.y + labelSize.y + 2), color, false, 2);

		//ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Description.ContextMenuLabel);
		//ImGui::RenderText(ImVec2(offsets->OffsetLabel + pos.x , pos.y), label);
		//ImGui::PopStyleColor();

		//if (shortcutWidth > 0.0f)
		//{
		//	ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Description.ContextMenuShortCut);
		//	ImGui::RenderText(ImVec2(offsets->OffsetShortcut + stretchWidth + pos.x + Description.ContextMenuShortcutWidth - shortcutWidth + 6, pos.y), shortcut, nullptr, false);
		//	ImGui::PopStyleColor();
		//}

		//if (selected) {
		//	ImGui::RenderCheckMark(window->DrawList, ImVec2(offsets->OffsetMark + stretchWidth + g.FontSize * 0.40f + pos.x, g.FontSize * 0.134f * 0.5f + pos.y), ImGui::GetColorU32(ImGuiCol_Text), g.FontSize * 0.866f);
		//}

		//IMGUI_TEST_ENGINE_ITEM_INFO(g.LastItemData.ID, label, g.LastItemData.StatusFlags | ImGuiItemStatusFlags_Checkable | (selected ? ImGuiItemStatusFlags_Checked : 0));

		//if (!enabled) {
		//	ImGui::EndDisabled();
		//}

		//ImGui::PopID();
		//if (menuSetOpen) {
		//	g.NavWindow = backed_nav_window;
		//}

		//return isPressed;
		return false;
	}

	void UI::ListBackground(ImU32 colorA, ImU32 colorB)
	{
		const float rowHeight = Description.TreeNodeHeight + 2.0f;
		ImDrawList* drawList = ImGui::GetWindowDrawList();
		ImVec2 windowSize = ImGui::GetWindowSize();
		ImVec2 cursorPos = ImGui::GetWindowPos();

		windowSize.x += rowHeight;
		cursorPos.y -= ImGui::GetScrollY();

		const uint32_t clippedCount = ImGui::GetScrollY() / 20 ;
		const uint32_t rowCount = windowSize.y / rowHeight + 2;

		cursorPos.y += clippedCount * rowHeight;
		s_ListColorCurrentIsDark = clippedCount % 2 == 0;

		for (uint32_t i = 0; i < rowCount; i++)
		{
			drawList->AddRectFilled(
				cursorPos, { cursorPos.x + windowSize.x, cursorPos.y + rowHeight },
				s_ListColorCurrentIsDark ? colorA : colorB);
			
			cursorPos.y += rowHeight;

			s_ListColorCurrentIsDark = !s_ListColorCurrentIsDark;
		}
	}

	void UI::ItemActivityOutline(const float rounding, const ImColor active, const ImColor inactive, const ImColor hovered)
	{
		ImDrawList* drawList = ImGui::GetWindowDrawList();
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
		//PushID();
		//ShiftCursorY(1.0f);

		//const bool layoutSuspended = []
		//{
		//	const ImGuiWindow* window = ImGui::GetCurrentWindow();
		//	if (window->DC.CurrentLayout)
		//	{
		//		ImGui::SuspendLayout();
		//		return true;
		//	}
		//	return false;
		//}();

		//bool modified = false;
		//bool searching = false;

		//const float areaPosX = ImGui::GetCursorPosX();
		//const float framePaddingY = ImGui::GetStyle().FramePadding.y;

		//ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
		//ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);
		//ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(22.0f, framePaddingY));
		//char searchBuffer[ENTITY_NAME_MAX_LENGTH]{};
		//strcpy_s<ENTITY_NAME_MAX_LENGTH>(searchBuffer, searchString.c_str());
		//ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImU32)Description.InputFieldBackground);

		//if (ImGui::InputText(GenerateID(), searchBuffer, ENTITY_NAME_MAX_LENGTH))
		//{
		//	searchString = searchBuffer;
		//	modified = true;
		//}
		//else if (ImGui::IsItemDeactivatedAfterEdit())
		//{
		//	searchString = searchBuffer;
		//	modified = true;
		//}

		//searching = searchBuffer[0] != 0;

		//if (grabFocus && *grabFocus)
		//{
		//	if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
		//		&& !ImGui::IsAnyItemActive()
		//		&& !ImGui::IsMouseClicked(0))
		//	{
		//		ImGui::SetKeyboardFocusHere(-1);
		//	}

		//	if (ImGui::IsItemFocused()) {
		//		*grabFocus = false;
		//	}
		//}

		//ItemActivityOutline(2.0f, Description.InputOutline, Description.InputOutline, Description.InputOutline);
		//ImGui::SetItemAllowOverlap();

		//ImGui::SameLine(areaPosX + 5.0f);

		//if (layoutSuspended) {
		//	ImGui::ResumeLayout();
		//}

		//ImGui::BeginHorizontal(GenerateID(), ImGui::GetItemRectSize());

		//// Search icon
		//{
		//	const float iconYOffset = framePaddingY - 1;
		//	ShiftCursorY(iconYOffset);
		//	Image(s_SearchIcon, ImVec2(s_SearchIcon->GetWidth(), s_SearchIcon->GetHeight()) , {1, 1, 1, 1});
		//	ShiftCursorX(4);
		//	ShiftCursorY(-iconYOffset);

		//	// Hint
		//	if (searching == false)
		//	{
		//		ShiftCursorY(-framePaddingY + 1.0f);
		//		ImGui::TextUnformatted(hint);
		//		ShiftCursorY(-1.0f);
		//	}
		//}

		//ImGui::Spring();

		//if (searching)
		//{
		//	const float spacingX = 4.0f;
		//	const float lineHeight = ImGui::GetItemRectSize().y - framePaddingY / 2.0f;

		//	if (ImGui::InvisibleButton(GenerateID(), ImVec2{ 18, 18 }))
		//	{
		//		searchString.clear();
		//		modified = true;
		//	}

		//	if (ImGui::IsMouseHoveringRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax())) {
		//		ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		//	}

		//	auto rect = UI::GetItemRect();
		//	rect.Min.y += 1;
		//	rect.Max.y += 1;

		//	UI::DrawButtonImage(s_CloseIcon, 
		//		Description.InputFieldClearButton,
		//		Description.InputFieldClearButtonHovered,
		//		Description.InputFieldClearButtonPressed,
		//		UI::RectExpanded(rect, -2.0f, -2.0f));
		//	ImGui::Spring(-1.0f, spacingX * 2.0f);
		//}

		//ImGui::PopStyleColor();
		//ImGui::EndHorizontal();
		//ImGui::PopStyleVar(3);

		//PopID();

		return false;
	}
}