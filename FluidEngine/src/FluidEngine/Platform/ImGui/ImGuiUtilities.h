#ifndef IMGUI_UTILITIES_H_
#define IMGUI_UTILITIES_H_

namespace fe {
	class UI
	{
	public:
		static void ShiftCursor(float x, float y);
		static void ShiftCursorX(float value);
		static void ShiftCursorY(float value);
	};
}

#endif // !IMGUI_UTILITIES_H_


