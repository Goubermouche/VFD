#ifndef IMGUI_UTILITIES_H
#define IMGUI_UTILITIES_H

#include <imgui_internal.h>
#include "Renderer/Texture.h"

namespace fe {
	class UI
	{
	public:
		static void ShiftCursor(float x, float y);
		static void ShiftCursorX(float value);
		static void ShiftCursorY(float value);

		static bool ItemHoverable(const ImRect& bb, ImGuiID id);

		static void Image(Ref<Texture> texture, const ImVec2& size);
		static void Image(Ref<Texture> texture, const ImVec2& size, const ImVec4& tintColor);
	};
}

#endif // !IMGUI_UTILITIES_H


