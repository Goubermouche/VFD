#ifndef KEY_EVENT_H
#define KEY_EVENT_H

#include "Event.h"
#include "Core/KeyCodes.h"

namespace fe {
	/// <summary>
	/// Base KeyEvent class
	/// </summary>
	class KeyEvent : public Event
	{
	public:
		KeyEvent(const KeyCode keycode)
			: m_KeyCode(keycode)
		{}

		/// <summary>
		/// Gets the KeyEvent's key code.
		/// </summary>
		/// <returns></returns>
		KeyCode GetKeyCode() const { 
			return m_KeyCode;
		}

		EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)
	protected:
		KeyCode m_KeyCode;
	};

	/// <summary>
	/// Called every time a key is pressed.
	/// </summary>
	class KeyPressedEvent : public KeyEvent
	{
	public:
		KeyPressedEvent(const KeyCode keycode, const int repeatCount)
			: KeyEvent(keycode), m_RepeatCount(repeatCount)
		{}

		int GetRepeatCount() const {
			return m_RepeatCount;
		}

		[[nodiscard]]
		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyPressedEvent: " << m_KeyCode << " (" << m_RepeatCount << " repeats)";
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyPressed)
	private:
		int m_RepeatCount;
	};

	/// <summary>
	/// Called every time a key is released.
	/// </summary>
	class KeyReleasedEvent : public KeyEvent
	{
	public:
		KeyReleasedEvent(KeyCode keycode)
			: KeyEvent(keycode)
		{}

		[[nodiscard]]
		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << m_KeyCode;
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyReleased)
	};

	/// <summary>
	/// Called every time a key is typed (only works with input fields).
	/// </summary>
	class KeyTypedEvent : public KeyEvent
	{
	public:
		KeyTypedEvent(KeyCode keycode)
			: KeyEvent(keycode)
		{}

		[[nodiscard]]
		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyTypedEvent: " << m_KeyCode;
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyTyped)
	};
}

#endif // !KEY_EVENT_H
