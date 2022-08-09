#ifndef UI_H
#define UI_H

#include <imgui_internal.h>
#include "Renderer/Texture.h"
#include "Editor/Editor.h"

namespace fe {
	struct UIDesc {

		ImColor Test = { 255, 0, 0, 255 };
		ImColor Transparent = { 0, 0, 0, 0 };

		// Lists
		ImColor ListBackgroundLight = { 43, 43, 43, 255 };
		ImColor ListBackgroundDark = { 40, 40, 40, 255 };
		ImColor ListToggleColor = { 194, 194, 194, 255 };
		ImColor ListTextColor = { 200, 200, 200, 255 };
		ImColor ListBackgroundHovered = { 68, 68, 68, 255 };
		ImColor ListSelectionActive = { 51, 77, 128, 255 };

		float ListRowHeight = 18.0f;

		// Input fields 
		ImColor InputFieldBackground = { 29, 29, 29, 255 };
		ImColor InputOutline = { 61, 61, 61, 255 };
		ImColor InputText = { 255, 255, 255, 255 };
		ImColor InputFieldClearButton = { 200, 200, 200, 255 };
		ImColor InputFieldClearButtonHovered = { 255, 255, 255, 255 };
		ImColor InputFieldClearButtonPressed = { 255, 255, 255, 255 };

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

		// Context menu
		ImColor ContextMenuBackground = { 24, 24, 24, 255 };
		ImColor ContextMenuLabel = { 255, 255, 255, 255 };
		ImColor ContextMenuShortCut = { 170, 170, 170, 255 };
		ImColor ContextMenuButtonBackground = { 24, 24, 24, 255 };
		ImColor ContextMenuButtonBackgroundHovered = { 71, 114, 179, 255 };
		ImColor ContextMenuArrow = { 194, 194, 194, 255 };
		ImColor ContextMenuBorder = { 54, 54, 54, 255 };

		float ContextMenuLabelWidth = 100.0f;
		float ContextMenuShortcutWidth = 50.0f;
		float ContextMenuIndent = 0.0f;

		// Frame time graph
		glm::vec3 FrameTimeGraphColors[4] = {
			{ 0,0, 1 }, // Blue
			{ 0,1, 0 }, // Green
			{ 1,1, 0 }, // Yellow
			{ 1,0, 0 }, // Red
		};
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
		static bool IsRootOfOpenMenuSet();

		static void Image(Ref<Texture> texture, const ImVec2& size);
		static void Image(Ref<Texture> texture, const ImVec2& size, const ImVec4& tintColor);

		static void DrawButtonImage(Ref<Texture> imageNormal, Ref<Texture> imageHovered, Ref<Texture> imagePressed,	ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed,	ImVec2 rectMin, ImVec2 rectMax);
		static void DrawButtonImage(Ref<Texture> texture, ImColor tintNormal, ImColor tintHovered, ImColor tintPressed, ImRect rectangle);
		static void DrawButtonImage(Ref<Texture> imageNormal, Ref<Texture> imageHovered, ImColor tintNormal, ImColor tintHovered, ImColor tintPressed, ImRect rectangle);

		static void Separator();

		static bool BeginMenu(const char* label, bool enabled = true);
		static bool MenuItem(const char* label, const char* shortcut = nullptr, bool selected = false, bool enabled = true);
		static void ListBackground();
		static void ItemActivityOutline(float rounding, ImColor active, ImColor inactive, ImColor hovered);

		class Widget {
		public:
			static bool SearchBar(std::string& searchString, const char* hint, bool* grabFocus = nullptr);

		private:
			static Ref<Texture> s_SearchIcon;
			static Ref<Texture> s_CloseIcon;

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

#endif // !UI_H