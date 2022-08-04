#ifndef MOUSE_EVENT_H_
#define MOUSE_EVENT_H_

#include "Event.h"
#include "Core/KeyCodes.h"

namespace fe {
	/// <summary>
	/// Called every time the mouse is moved.
	/// </summary>
	class MouseMovedEvent : public Event
	{
	public:
		MouseMovedEvent(const float x, const float y)
			: m_MouseX(x), m_MouseY(y)
		{}

		[[nodiscard]]
		float GetX() const { 
			return m_MouseX; 
		}

		[[nodiscard]]
		float GetY() const { 
			return m_MouseY; 
		}

		[[nodiscard]]
		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "MouseMovedEvent: " << m_MouseX << ", " << m_MouseY;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseMoved)
		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	private:
		float m_MouseX;
		float m_MouseY;
	};

	/// <summary>
	/// Called every time the mouse scroll wheel moves.
	/// </summary>
	class MouseScrolledEvent : public Event
	{
	public:
		MouseScrolledEvent(const float xOffset, const float yOffset)
			: m_XOffset(xOffset), m_YOffset(yOffset)
		{}

		[[nodiscard]]
		float GetXOffset() const { 
			return m_XOffset; 
		}

		[[nodiscard]]
		float GetYOffset() const {
			return m_YOffset;
		}

		[[nodiscard]]
		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "MouseScrolledEvent: " << GetXOffset() << ", " << GetYOffset();
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseScrolled)
		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	private:
		float m_XOffset;
		float m_YOffset;
	};

	/// <summary>
	/// Base mouse button class.
	/// </summary>
	class MouseButtonEvent : public Event
	{
	public:
		MouseButtonEvent(const MouseButton button)
			: m_Button(button)
		{}

		[[nodiscard]]
		MouseButton GetMouseButton() const { 
			return m_Button; 
		}

		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	protected:
		MouseButton m_Button;
	};

	/// <summary>
	/// Called every time a mouse button is pressed.
	/// </summary>
	class MouseButtonPressedEvent : public MouseButtonEvent
	{
	public:
		MouseButtonPressedEvent(const MouseButton button)
			: MouseButtonEvent(button)
		{}

		[[nodiscard]]
		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "MouseButtonPressedEvent: " << m_Button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseButtonPressed)
	};

	/// <summary>
	/// Called every time a mouse button is released.
	/// </summary>
	class MouseButtonReleasedEvent : public MouseButtonEvent
	{
	public:
		MouseButtonReleasedEvent(const MouseButton button)
			: MouseButtonEvent(button)
		{}

		[[nodiscard]]
		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "MouseButtonReleasedEvent: " << m_Button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseButtonReleased)
	};
}

#endif // !MOUSE_EVENT_H_