#include "pch.h"
#include "UI.h"

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

	void UI::Image(Ref<Texture> texture, const ImVec2& size)
	{
		ImGui::Image((void*)(intptr_t)texture->GetRendererID(), size);
	}

	void UI::Image(Ref<Texture> texture, const ImVec2& size, const ImVec4& tintColor)
	{
		ImGui::Image((void*)(intptr_t)texture->GetRendererID(), size, ImVec2{ 0, 0 }, ImVec2{ 1, 1 }, tintColor);
	}
}