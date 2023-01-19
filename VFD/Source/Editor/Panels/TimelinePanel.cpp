#include "pch.h"
#include "TimelinePanel.h"

#include "Core/Time.h"

namespace vfd
{
	TimelinePanel::TimelinePanel()
	{
		
	}

	void TimelinePanel::OnUpdate()
	{
		constexpr float grabPadding = 4.0f;
		constexpr float animationMargin = 4.0f;
		float animationHeight = 100.0f;

		unsigned int maxFrames = 100u;
		unsigned int animCount = 0u;
		const float deltaFrameIndex = m_FrameIndex;


		for (const entt::entity entity : m_SceneContext->View<DFSPHSimulationComponent>()) {
			Entity e = { entity, m_SceneContext.Raw() };
			auto& simulation = e.GetComponent<DFSPHSimulationComponent>();

			DFSPHSimulationDescription desc = simulation.Handle->GetDescription();

			if (desc.FrameCount > maxFrames)
			{
				maxFrames = desc.FrameCount;
			}

			animCount++;
		}


		ImVec2 avail = ImGui::GetContentRegionAvail();
		ImVec2 windowPosition = ImGui::GetWindowPos();
		ImDrawList* drawList = ImGui::GetWindowDrawList();
		windowPosition.y += 21 + 20;
		avail.y -= 20;

		// Animations
		if(animationHeight * animCount > avail.y)
		{
			animationHeight = (avail.y - animationMargin * animCount) / animCount + animationMargin;
		}

		unsigned int animationIndex = 0u;
		for (const entt::entity entity : m_SceneContext->View<DFSPHSimulationComponent>()) {
			Entity e = { entity, m_SceneContext.Raw() };
			auto& simulation = e.GetComponent<DFSPHSimulationComponent>();

			DFSPHSimulationDescription desc = simulation.Handle->GetDescription();
			DFSPHDebugInfo info = simulation.Handle->GetDebugInfo();

			const unsigned int frameCount = desc.FrameCount;
			const float widthRatio = static_cast<float>(frameCount) / static_cast<float>(maxFrames);

			const ImVec2 min = {
				windowPosition.x,
				windowPosition.y + animationHeight * animationIndex + animationMargin
			};

			const ImVec2 max = {
				windowPosition.x + widthRatio * avail.x,
				windowPosition.y + animationHeight * animationIndex + animationHeight
			};

			drawList->AddRectFilled(
				min,
				max,
				IM_COL32(17.0f, 26.0f, 38.0f, 255.0f)
			);

			// Name
			ImGui::PushClipRect(min, max, false);
			// ImGui::SetWindowFontScale(1.3f);

			drawList->AddText(
				{ min.x + 6.0f, min.y + 2.0f },
				IM_COL32(255.0f, 255.0f, 255.0f, 255.0f),
				e.GetComponent<TagComponent>().Tag.c_str()
			);

			ImGui::PopClipRect();
			// ImGui::SetWindowFontScale(1.0f);

			animationIndex++;
		}

		windowPosition = ImGui::GetWindowPos();
		windowPosition.y += 21;

		avail = ImGui::GetContentRegionAvail();

		ImGui::SetNextItemWidth(avail.x);

		if (m_Paused == false)
		{
			m_FrameIndex += 60.0f * Time::GetDeltaTime();

			if(m_FrameIndex > maxFrames)
			{
				m_FrameIndex = 0.0f;
			}
			m_CursorPositionTimeline = m_FrameIndex / static_cast<float>(maxFrames);
		}

		// Slider grab
		const std::string frameIndexText = std::to_string(static_cast<int>(m_FrameIndex));
		const float frameIndexTextWidth = ImGui::CalcTextSize(frameIndexText.c_str()).x;

		const float grabWidth = frameIndexTextWidth + grabPadding * 2;
		avail.x -= grabWidth + 4.0f;

		ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, grabWidth);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(50.0f, 50.0f, 50.0f, 255.0f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, IM_COL32(50.0f, 50.0f, 50.0f, 255.0f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, IM_COL32(50.0f, 50.0f, 50.0f, 255.0f));

		if(m_Paused)
		{
			ImGui::SliderFloat("##", &m_CursorPositionTimeline, 0.0f, 1.0f, "");
			m_FrameIndex = maxFrames * m_CursorPositionTimeline;
		}
		else
		{
			float fake = m_CursorPositionTimeline;
			ImGui::SliderFloat("##", &fake, 0.0f, 1.0f, "");

			if(fake != m_CursorPositionTimeline)
			{
				m_CursorPositionTimeline = fake;
				m_FrameIndex = m_CursorPositionTimeline * maxFrames;
			}
		}

		ImGui::PopStyleVar();
		ImGui::PopStyleColor(3);

		// Slider line
		const float caretXPos = avail.x * m_CursorPositionTimeline;

		drawList->AddLine(
			{ windowPosition.x + caretXPos + grabWidth / 2.0f + 2.0f, windowPosition.y + 18.0f },
			{ windowPosition.x + caretXPos + grabWidth / 2.0f + 2.0f, windowPosition.y + avail.y },
			IM_COL32(61.0f, 133.0f, 224.0f, 255.0f), 3.0f
		);

		// Frame index
		drawList->AddText(
			ImVec2(windowPosition.x + caretXPos + 2.0f + grabPadding, windowPosition.y + 3.0f),
			IM_COL32(255.0f, 255.0f, 255.0f, 255.0f),
			frameIndexText.c_str()
		);

		if(static_cast<unsigned int>(deltaFrameIndex) != static_cast<unsigned int>(m_FrameIndex))
		{
			for (const entt::entity entity : m_SceneContext->View<DFSPHSimulationComponent>()) {
				Entity e = { entity, m_SceneContext.Raw() };
				auto& simulation = e.GetComponent<DFSPHSimulationComponent>();

				Ref<DFSPHParticleBuffer> buffer = simulation.Handle->GetParticleFrameBuffer();

				if(buffer)
				{
					buffer->SetActiveFrame(static_cast<unsigned int>(m_FrameIndex));
				}
			}
		}
	}

	void TimelinePanel::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<KeyPressedEvent>(BIND_EVENT_FN(OnKeyPressed));
	}

	bool TimelinePanel::OnKeyPressed(KeyPressedEvent& event)
	{
		// Call shortcuts only when the key get pressed.
		if (event.GetRepeatCount() > 0) {
			return false;
		}

		if(event.GetKeyCode() == KeyCode::Space)
		{
			m_Paused = !m_Paused;
		}

		return false;
	}
}