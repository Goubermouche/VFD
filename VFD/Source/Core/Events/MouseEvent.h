#ifndef MOUSE_EVENT_H
#define MOUSE_EVENT_H

#include "Event.h"
#include "Core/KeyCodes.h"

namespace vfd {
	/// <summary>
	/// [Event] Called every time the mouse is moved.
	/// </summary>
	class MouseMovedEvent : public Event
	{
	public:
		MouseMovedEvent(const float x, const float y)
			: m_MouseX(x), m_MouseY(y)
		{}

		~MouseMovedEvent() override = default;

		[[nodiscard]]
		float GetX() const { 
			return m_MouseX; 
		}

		[[nodiscard]]
		float GetY() const { 
			return m_MouseY; 
		}

		[[nodiscard]]
		const glm::vec2& Get() const {
			return { m_MouseX, m_MouseY };
		}

		[[nodiscard]]
		const std::string& ToString() const override
		{
			std::stringstream ss;
			ss << "MouseMovedEvent: " << m_MouseX << ", " << m_MouseY;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseMoved)
		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	private:
		float m_MouseX = 0.0f;
		float m_MouseY = 0.0f;
	};

	/// <summary>
	/// [Event] Called every time the mouse scroll wheel moves.
	/// </summary>
	class MouseScrolledEvent : public Event
	{
	public:
		MouseScrolledEvent(const float xOffset, const float yOffset)
			: m_OffsetX(xOffset), m_OffsetY(yOffset)
		{}

		~MouseScrolledEvent() override = default;

		[[nodiscard]]
		float GetXOffset() const { 
			return m_OffsetX; 
		}

		[[nodiscard]]
		float GetYOffset() const {
			return m_OffsetY;
		}

		[[nodiscard]]
		const std::string& ToString() const override
		{
			std::stringstream ss;
			ss << "MouseScrolledEvent: " << GetXOffset() << ", " << GetYOffset();
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseScrolled)
		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	private:
		float m_OffsetX = 0.0f;
		float m_OffsetY = 0.0f;
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

		~MouseButtonEvent() override = default;

		[[nodiscard]]
		MouseButton GetMouseButton() const { 
			return m_Button; 
		}

		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	protected:
		MouseButton m_Button;
	};

	/// <summary>
	/// [Event] Called every time a mouse button is pressed.
	/// </summary>
	class MouseButtonPressedEvent : public MouseButtonEvent
	{
	public:
		MouseButtonPressedEvent(const MouseButton button)
			: MouseButtonEvent(button)
		{}

		~MouseButtonPressedEvent() override = default;

		[[nodiscard]]
		const std::string& ToString() const override
		{
			std::stringstream ss;
			ss << "MouseButtonPressedEvent: " << m_Button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseButtonPressed)
	};

	/// <summary>
	/// [Event] Called every time a mouse button is released.
	/// </summary>
	class MouseButtonReleasedEvent : public MouseButtonEvent
	{
	public:
		MouseButtonReleasedEvent(const MouseButton button)
			: MouseButtonEvent(button)
		{}

		~MouseButtonReleasedEvent() override = default;

		[[nodiscard]]
		const std::string& ToString() const override
		{
			std::stringstream ss;
			ss << "MouseButtonReleasedEvent: " << m_Button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseButtonReleased)
	};
}

#endif // !MOUSE_EVENT_H