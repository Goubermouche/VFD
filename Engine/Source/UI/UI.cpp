#include "pch.h"
#include "UI.h"

#include <imgui.h>
#include "UI/ImGui/ImGuiRenderer.h" 
#include "UI/ImGui/ImGuiGLFWBackend.h"
#include "Core/Application.h"

namespace fe {
	UIDesc UI::Description;
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

		// TODO: move to UI::Init()
		// Initialize the ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		// TODO: Create a separate UI context so that support for other platforms can be added (?) - not important right now
		ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow()), true);
		ImGui_ImplOpenGL3_Init("#version 410"); // Use GLSL version 410

		// IO
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigWindowsMoveFromTitleBarOnly = true;

		// Font
		ImFontConfig config;
		config.OversampleH = 5;
		config.OversampleV = 5;
		static const ImWchar ranges[] =
		{
			0x0020, 0x00FF, // Basic Latin + Latin Supplement
			0x2200, 0x22FF, // Mathematical Operators
			0x0370, 0x03FF, // Greek and Coptic
			0,
		};

		io.FontDefault = io.Fonts->AddFontFromFileTTF("Resources/Fonts/OpenSans/OpenSans-SemiBold.ttf", 15.0f, &config, ranges);

		// Style
		ImGui::StyleColorsDark();
		ImGuiStyle& style = ImGui::GetStyle();
		style.ItemSpacing = { 0.0f, 0.0f };
		style.WindowPadding = { 0.0f, 0.0f };
		style.ScrollbarRounding = 2.0f;
		style.FrameBorderSize = 1.0f;
		style.TabRounding = 0.0f;
		style.WindowMenuButtonPosition = ImGuiDir_None;
		style.WindowRounding = 2.0f;
		style.WindowMinSize = { 100.0f, 109.0f };
		style.WindowBorderSize = 0;
		style.ChildBorderSize = 0;
		style.FrameBorderSize = 0;

		style.Colors[ImGuiCol_WindowBg] = Description.WindowBackground;
		style.Colors[ImGuiCol_TitleBg] = Description.WindowTitleBackground;
		style.Colors[ImGuiCol_TitleBgActive] = Description.WindowTitleBackgroundFocused;
		
		style.Colors[ImGuiCol_Tab] = Description.TabBackground;
		style.Colors[ImGuiCol_TabUnfocused] = Description.TabBackground;
		style.Colors[ImGuiCol_TabUnfocusedActive] = Description.TabBackground;
		style.Colors[ImGuiCol_TabHovered] = Description.TabBackgroundHovered;
		style.Colors[ImGuiCol_TabActive] = Description.TabBackgroundFocused;

		style.Colors[ImGuiCol_Separator] = Description.Separator;
		style.Colors[ImGuiCol_SeparatorActive] = Description.Separator;
		style.Colors[ImGuiCol_SeparatorHovered] = Description.Separator;


		LOG("ImGui initialized successfully", "editor][ImGui");
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

	void UI::TreeBackground()
	{
		float rowHeight = Description.RowHeight + 2.0f;
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
				s_ListColorCurrentIsDark ? Description.ListBackgroundDark : Description.ListBackgroundLight);
			
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
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImU32)Description.InputFieldBackground);

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

		UI::ItemActivityOutline(2.0f, Description.InputOutline, Description.InputOutline, Description.InputOutline);
		ImGui::SetItemAllowOverlap();

		ImGui::SameLine(areaPosX + 5.0f);


		if (layoutSuspended) {
			ImGui::ResumeLayout();
		}

		ImGui::BeginHorizontal(GenerateID(), ImGui::GetItemRectSize());

		// Search icon
		{
			const float iconYOffset = framePaddingY - 1;
			UI::ShiftCursorY(iconYOffset);
			UI::Image(s_SearchIcon, ImVec2(s_SearchIcon->GetWidth(), s_SearchIcon->GetHeight()) , {1, 1, 1, 1});
			UI::ShiftCursorX(4);
			UI::ShiftCursorY(-iconYOffset);

			// Hint
			if (searching == false)
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