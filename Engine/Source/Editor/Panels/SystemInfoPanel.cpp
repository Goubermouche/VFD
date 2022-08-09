#include "pch.h"
#include "SystemInfoPanel.h"

#include "Core/Time.h"
#include "Compute/GPUCompute.h"
#include "Core/Application.h"
#include "UI/UI.h"

namespace fe {
	SystemInfoPanel::SystemInfoPanel()
	{
	}

	glm::vec4 SystemInfoPanel::CalculateDeltaTimeColor(const float dt) const
	{
		constexpr uint32_t colorCount = 4;

		const float times[colorCount] = {
			1.0f / m_FramesThresholdBlue ,
			1.0f / m_FramesThresholdGreen ,
			1.0f / m_FramesThresholdYellow ,
			1.0f / m_FramesThresholdRed,
		};

		if (dt < times[0]) {
			return glm::vec4(UI::Description.FrameTimeGraphColors[0], 1.f);
		}

		for (size_t i = 1; i < colorCount; ++i)
		{
			if (dt < times[i])
			{
				const float t = (dt - times[i - 1]) / (times[i] - times[i - 1]);
				return glm::vec4(glm::mix(UI::Description.FrameTimeGraphColors[i - 1], UI::Description.FrameTimeGraphColors[i], t), 1.0f);
			}
		}

		return glm::vec4(UI::Description.FrameTimeGraphColors[colorCount - 1], 1.f);
	}

	void SystemInfoPanel::OnUpdate()
	{
		const bool VSync = Application::Get().GetWindow().IsVSync();
		m_FrameTimeHistory.AddEntry(Time::GetDeltaTime());

		const float width = ImGui::GetWindowWidth();
		const uint32_t frameCount = m_FrameTimeHistory.GetCount();

		UI::ShiftCursor(2, 2);
		// Info
		ImGui::Text(
			"%0.f FPS (%0.3f ms) (VSync: %s) (DrawCalls: %d)",
			1.0f / Time::GetDeltaTime(), 
			Time::GetDeltaTime() * 1000.0f, 
			VSync ? "on" : "off",
			Renderer::GetDrawCallCount()
		);

		UI::ShiftCursorY(2);

		// Frame time graph
		if (width > 0.f && frameCount > 0)
		{
			if(VSync)
			{
				// VSync on
				// TODO: Potentially get the monitor refresh rate and use that instead of 60. 
				m_FramesThresholdBlue = 55;
				m_FramesThresholdGreen = 50;
				m_FramesThresholdYellow = 30;
				m_FramesThresholdRed = 15;
			}
			else
			{
				// VSync on
				m_FramesThresholdBlue = 1500;
				m_FramesThresholdGreen = 1000;
				m_FramesThresholdYellow = 200;
				m_FramesThresholdRed = 75;
			}

			ImDrawList* drawList = ImGui::GetWindowDrawList();
			const ImVec2 cursorPos = ImGui::GetCursorScreenPos();
			float position = width;

			const float deltaTimeMin = 1.0f / m_FramesThresholdBlue;
			const float deltaTimeMax = 1.0f / m_FramesThresholdRed;
			const float deltaTimeMinLog2 = std::log2(deltaTimeMin);
			const float deltaTimeMaxLog2 = std::log2(deltaTimeMax);

			drawList->AddRectFilled(cursorPos, ImVec2(cursorPos.x + width, cursorPos.y + m_FrameGraphMaxHeight), UI::Description.ListBackgroundDark);

			for (uint32_t frameIndex = 0; frameIndex < frameCount && position > 0.0f; ++frameIndex)
			{
				const FrameTimeHistory::Entry dt = m_FrameTimeHistory.GetEntry(frameIndex);
				const float frameWidth = dt.DeltaTime / deltaTimeMin;

				const float frameHeightFactor = (dt.DeltaTimeLog2 - deltaTimeMinLog2) / (deltaTimeMaxLog2 - deltaTimeMinLog2);
				const float frameHeightFactor_Nrm = std::min(std::max(0.0f, frameHeightFactor), 1.0f);
				const float frameHeight = glm::mix(m_FrameGraphMinHeight, m_FrameGraphMaxHeight, frameHeightFactor_Nrm);;

				const ImVec2 min = { cursorPos.x+  position - frameWidth,  cursorPos.y + m_FrameGraphMaxHeight - frameHeight };
				const ImVec2 max = { cursorPos.x + position, cursorPos.y + m_FrameGraphMaxHeight };

				const uint32_t color = glm::packUnorm4x8(CalculateDeltaTimeColor(dt.DeltaTime));

				drawList->AddRectFilled(min, max, color	);

				position -= frameWidth;
			}

			ImGui::Dummy(ImVec2(width, m_FrameGraphMaxHeight));
		}
	}

	FrameTimeHistory::Entry FrameTimeHistory::GetEntry(uint32_t index) const
	{
		index = (m_Back + m_Count - index - 1) % s_Capacity;
		return m_Entries[index];
	}

	void FrameTimeHistory::AddEntry(const float dt)
	{
		m_Entries[m_Front] = { dt, log2(dt) };
		m_Front = (m_Front + 1) % s_Capacity;

		if (m_Count == s_Capacity) {
			m_Back = m_Front;
		}
		else {
			m_Count++;
		}
	}
}
