#include "pch.h"
#include "ImGuiUtilities.h"
#include <imgui.h>


namespace fe {
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
}