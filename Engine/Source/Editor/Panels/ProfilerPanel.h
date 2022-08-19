#ifndef SYSTEM_INFO_PANEL_H
#define SYSTEM_INFO_PANEL_H

#include "Editor/Panels/EditorPanel.h"

namespace fe {
	class FrameTimeHistory {
	public:
		FrameTimeHistory() = default;
		~FrameTimeHistory() = default;

		struct Entry {
			float DeltaTime;
			float DeltaTimeLog2;
		};

		[[nodiscard]]
		uint32_t GetCount() const {
			return m_Count;
		}

		void Clear() { 
			*this = {};
		}

		[[nodiscard]]
		Entry GetEntry(uint32_t index) const;
		void AddEntry(float deltaTime);
	private:
		static constexpr uint32_t s_Capacity = 512;
		uint32_t m_Back = 0;
		uint32_t m_Front = 0;
		uint32_t m_Count = 0;
		Entry m_Entries[s_Capacity];
	};

	class ProfilerPanel : public EditorPanel
	{
	public:
		ProfilerPanel() = default;
		~ProfilerPanel() override = default;

		void OnUpdate() override;
	private:
		glm::vec4 CalculateDeltaTimeColor(float deltaTime) const;
	private:
		FrameTimeHistory m_FrameTimeHistory;

		float m_MinFrameTime = FLT_MAX;
		float m_MaxFrameTime = FLT_MIN;

		const float m_FrameGraphMinHeight = 8.0f;
		const float m_FrameGraphMaxHeight = 48.0f;
		static constexpr float m_FrameGraphOffset = 1.0f;

		float m_FramesThresholdBlue;
		float m_FramesThresholdGreen;
		float m_FramesThresholdYellow;
		float m_FramesThresholdRed;
	};
}

#endif // !SYSTEM_INFO_PANEL_H