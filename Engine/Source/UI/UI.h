#ifndef IMGUI_UTILITIES_H
#define IMGUI_UTILITIES_H

#include <imgui_internal.h>
#include "Renderer/Texture.h"
#include "Editor/Editor.h"

namespace fe {
	struct UIDesc {
		float RowHeight = 18.0f;

		// Colors
		ImColor Test = { 255, 0, 0, 255 };

		// Lists
		ImColor ListBackgroundLight = { 43, 43, 43, 255 };
		ImColor ListBackgroundDark = { 40, 40, 40, 255 };
		ImColor ListToggleColor = { 194, 194, 194, 255 };
		ImColor ListTextColor = { 200, 200, 200, 255 };

		ImColor ListBackgroundHovered = { 68, 68, 68, 255 };

		ImColor SelectionActive = { 51, 77, 128, 255 };

		// Input fields 
		ImColor InputFieldBackground = { 29, 29, 29, 255 };
		ImColor InputOutline = { 61, 61, 61, 255 };
		ImColor InputText = { 255, 255, 255, 255 };

		// Windows (Panels)
		ImColor WindowBackground = { 48, 48, 48, 255};
		ImColor WindowTitleBackground = { 29, 29, 29, 255 };
		ImColor WindowTitleBackgroundFocused = { 29, 29, 29, 255 };

		// Tabs
		ImColor TabBackground = { 29, 29, 29, 255 };
		ImColor TabBackgroundHovered = { 48, 48, 48, 255 };
		ImColor TabBackgroundFocused = { 48, 48, 48, 255 };

		// Separator
		ImColor Separator = { 22, 22, 22, 255 };
	};

	class UI
	{
	public:
		static void Init();

		static void ShiftCursor(float x, float y);
		static void ShiftCursorX(float value);
		static void ShiftCursorY(float value);

		static const char* GenerateID();
		static void PushID();
		static void PopID();

		static bool ItemHoverable(const ImRect& bb, ImGuiID id);
		static inline ImRect RectExpanded(const ImRect& rect, float x, float y);
		static inline ImRect GetItemRect();

		static void Image(Ref<Texture> texture, const ImVec2& size);
		static void Image(Ref<Texture> texture, const ImVec2& size, const ImVec4& tintColor);

		static void TreeBackground();

		static void ItemActivityOutline(float rounding, ImColor active, ImColor inactive, ImColor hovered);

		class Widget {
		public:
			static bool SearchBar(std::string& searchString, const char* hint, bool* grabFocus = nullptr);

		private:
			static Ref<Texture> s_SearchIcon;

			friend class UI;
		};

	public:
		static UIDesc Description;
	private:
		static bool s_ListColorCurrentIsDark;

		// ids
		static int s_UIContextID;
		static uint32_t s_Counter;
		static char s_IDBuffer[16];
	};
}

#endif // !IMGUI_UTILITIES_H