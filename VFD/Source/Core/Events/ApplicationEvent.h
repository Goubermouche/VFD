#ifndef APPLICATION_EVENT_H
#define APPLICATION_EVENT_H

#include "Event.h"

namespace vfd {
	/// <summary>
	/// [Event] Called every time the window is resized.
	/// </summary>
	class WindowResizeEvent : public Event
	{
	public:
		WindowResizeEvent(const uint16_t width, const uint16_t height)
			: m_Width(width), m_Height(height)
		{}

		~WindowResizeEvent() override = default;

		[[nodiscard]]
		uint16_t GetWidth() const { 
			return m_Width; 
		}

		[[nodiscard]]
		uint16_t GetHeight() const {
			return m_Height; 
		}

		[[nodiscard]]
		const glm::ivec2& Get() const {
			return { m_Width, m_Height };
		}

		[[nodiscard]]
		const std::string& ToString() const override
		{
			std::stringstream ss;
			ss << "WindowResizeEvent: " << m_Width << ", " << m_Height;
			return ss.str();
		}

		EVENT_CLASS_TYPE(WindowResize)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	private:
		uint16_t m_Width = 0;
		uint16_t m_Height = 0;
	};

	/// <summary>
	/// [Event] Called every time the window is minimized.
	/// </summary>
	class WindowMinimizeEvent : public Event
	{
	public:
		WindowMinimizeEvent(const bool minimized)
			: m_Minimized(minimized)
		{}

		~WindowMinimizeEvent() override = default;

		[[nodiscard]]
		bool IsMinimized() const { 
			return m_Minimized;
		}

		EVENT_CLASS_TYPE(WindowMinimize)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	private:
		bool m_Minimized = false;
	};

	/// <summary>
	/// [Event] Called once the window is closed.
	/// </summary>
	class WindowCloseEvent : public Event
	{
	public:
		WindowCloseEvent() = default;
		~WindowCloseEvent() override = default;

		EVENT_CLASS_TYPE(WindowClose)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};
}

#endif // !APPLICATION_EVENT_H
