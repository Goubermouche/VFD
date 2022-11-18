#ifndef UI_H
#define UI_H

#include <imgui_internal.h>
#include "Renderer/Texture.h"
#include "Editor/Editor.h"

namespace vfd {
	struct UIDesc {

		ImColor Test = { 255, 0, 0, 255 };
		ImColor Transparent = { 0, 0, 0, 0 };

		// Windows (Panels)
		ImColor WindowBackground = { 10, 10, 10, 255};
		ImColor WindowTitleBackground = { 29, 29, 29, 255 };
		ImColor WindowTitleBackgroundFocused = { 29, 29, 29, 255 };

		// Frame time graph
		glm::vec3 FrameTimeGraphColors[4] = {
			{ 0.0f, 0.0f, 1.0f }, // Blue
			{ 0.0f, 1.0f, 0.0f }, // Green
			{ 1.0f, 1.0f, 0.0f }, // Yellow
			{ 1.0f, 0.0f, 0.0f }, // Red
		};

		float TreeNodeHeight = 14.0f; // 14
		float TreeNodeTextOffsetY = 1.0f; // 1
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

		static void Image(Ref<Texture> texture, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1));
		static void Image(Ref<Texture> texture, const ImVec2& size, const ImVec4& tintColor);

		static void DrawButtonImage(Ref<Texture> imageNormal, Ref<Texture> imageHovered, Ref<Texture> imagePressed,	ImU32 tintNormal, ImU32 tintHovered, ImU32 tintPressed,	ImVec2 rectMin, ImVec2 rectMax);
		static void DrawButtonImage(Ref<Texture> texture, ImColor tintNormal, ImColor tintHovered, ImColor tintPressed, ImRect rectangle);
		static void DrawButtonImage(Ref<Texture> imageNormal, Ref<Texture> imageHovered, ImColor tintNormal, ImColor tintHovered, ImColor tintPressed, ImRect rectangle);

		static void Separator();

		static bool BeginMenu(const char* label, bool enabled = true);
		static bool MenuItem(const char* label, const char* shortcut = nullptr, bool selected = false, bool enabled = true);
		static void ListBackground(ImU32 colorA, ImU32 colorB);
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