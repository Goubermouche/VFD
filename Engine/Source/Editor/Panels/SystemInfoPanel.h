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
		void AddEntry(float dt);
	private:
		static constexpr uint32_t s_Capacity = 1024;
		uint32_t m_Back = 0;
		uint32_t m_Front = 0;
		uint32_t m_Count = 0;
		Entry m_Entries[s_Capacity];
	};

	class SystemInfoPanel : public EditorPanel
	{
	public:
		SystemInfoPanel();
		~SystemInfoPanel() override = default;

		void OnUpdate() override;
	private:
		glm::vec4 CalculateDeltaTimeColor(float dt) const;
	private:
		FrameTimeHistory m_FrameTimeHistory;

		float m_MinFrameTime = std::numeric_limits<float>::max();
		float m_MaxFrameTime = std::numeric_limits<float>::min();

		const float m_FrameGraphMinHeight = 8.0f;
		const float m_FrameGraphMaxHeight = 68.0f;
		static constexpr float m_FrameGraphOffset = 1.0f;

		float m_FramesThresholdBlue;
		float m_FramesThresholdGreen;
		float m_FramesThresholdYellow;
		float m_FramesThresholdRed;
	};
}

#endif // !SYSTEM_INFO_PANEL_H