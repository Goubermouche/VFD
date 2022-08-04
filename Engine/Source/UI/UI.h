#ifndef IMGUI_UTILITIES_H_
#define IMGUI_UTILITIES_H_

#include <imgui_internal.h>

namespace fe {
	class UI
	{
	public:
		static void ShiftCursor(float x, float y);
		static void ShiftCursorX(float value);
		static void ShiftCursorY(float value);

		static bool ItemHoverable(const ImRect& bb, ImGuiID id);
	};
}

#endif // !IMGUI_UTILITIES_H_


