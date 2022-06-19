#ifndef APPLICATION_EVENT_H_
#define APPLICATION_EVENT_H_

#include "Event.h"

namespace fe {
	/// <summary>
	/// Called every time the window is resized.
	/// </summary>
	class WindowResizeEvent : public Event
	{
	public:
		WindowResizeEvent(unsigned int width, unsigned int height)
			: m_Width(width), m_Height(height) {}

		inline unsigned int GetWidth() const { 
			return m_Width; 
		}

		inline unsigned int GetHeight() const {
			return m_Height; 
		}

		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "WindowResizeEvent: " << m_Width << ", " << m_Height;
			return ss.str();
		}

		EVENT_CLASS_TYPE(WindowResize)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	private:
		unsigned int m_Width, m_Height;
	};

	/// <summary>
	/// Called every time the window is minimized.
	/// </summary>
	class WindowMinimizeEvent : public Event
	{
	public:
		WindowMinimizeEvent(bool minimized)
			: m_Minimized(minimized) {}

		bool IsMinimized() const { 
			return m_Minimized;
		}

		EVENT_CLASS_TYPE(WindowMinimize)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	private:
		bool m_Minimized = false;
	};

	/// <summary>
	/// Called once the window is closed.
	/// </summary>
	class WindowCloseEvent : public Event
	{
	public:
		WindowCloseEvent() {}

		EVENT_CLASS_TYPE(WindowClose)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};
}

#endif // !APPLICATION_EVENT_H_
