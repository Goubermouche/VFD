#ifndef APPLICATION_EVENT_H_
#define APPLICATION_EVENT_H_

#include "Event.h"

namespace fe {
	// TODO: Should this store previous size?
	class WindowResizeEvent : public Event
	{
	public:
		WindowResizeEvent(unsigned int width, unsigned int height)
			: m_Width(width), m_Height(height) {}

		inline unsigned int GetWidth() const { return m_Width; }
		inline unsigned int GetHeight() const { return m_Height; }

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

	class WindowMinimizeEvent : public Event
	{
	public:
		WindowMinimizeEvent(bool minimized)
			: m_Minimized(minimized) {}

		bool IsMinimized() const { return m_Minimized; }

		EVENT_CLASS_TYPE(WindowMinimize)
			EVENT_CLASS_CATEGORY(EventCategoryApplication)
	private:
		bool m_Minimized = false;
	};

	class WindowCloseEvent : public Event
	{
	public:
		WindowCloseEvent() {}

		EVENT_CLASS_TYPE(WindowClose)
			EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};

	class WindowTitleBarHitTestEvent : public Event
	{
	public:
		WindowTitleBarHitTestEvent(int x, int y, int& hit)
			: m_X(x), m_Y(y), m_Hit(hit) {}

		inline int GetX() const { return m_X; }
		inline int GetY() const { return m_Y; }
		inline void SetHit(bool hit) { m_Hit = (int)hit; }

		EVENT_CLASS_TYPE(WindowTitleBarHitTest)
			EVENT_CLASS_CATEGORY(EventCategoryApplication)
	private:
		int m_X;
		int m_Y;
		int& m_Hit;
	};
}

#endif // !APPLICATION_EVENT_H_
