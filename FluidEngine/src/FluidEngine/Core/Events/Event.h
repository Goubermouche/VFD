#ifndef EVENT_H_
#define EVENT_H_

#include "pch.h"

namespace fe {
	enum class EventType
	{
		None = 0,
		WindowClose, WindowMinimize, WindowResize, WindowFocus, WindowLostFocus, WindowMoved, WindowTitleBarHitTest,
		KeyPressed, KeyReleased, KeyTyped,
		MouseButtonPressed, MouseButtonReleased, MouseMoved, MouseScrolled
	};

// Bit shifts the specified variable.
#define BIT(x) (1u << x)

	enum EventCategory
	{
		None = 0,
		EventCategoryApplication = BIT(0),
		EventCategoryInput = BIT(1),
		EventCategoryKeyboard = BIT(2),
		EventCategoryMouse = BIT(3),
		EventCategoryMouseButton = BIT(4)
	};

#define EVENT_CLASS_TYPE(type) static EventType GetStaticType() { return EventType::##type; }\
								virtual EventType GetEventType() const override { return GetStaticType(); }\
								virtual const char* GetName() const override { return #type; }

#define EVENT_CLASS_CATEGORY(category) virtual int GetCategoryFlags() const override { return category; }

// Binds function to a specific event.
#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }

	/// <summary>
	/// Base event class.
	/// </summary>
	class Event
	{
	public:
		bool Handled = false;

		/// <summary>
		/// Gets the event type.
		/// </summary>
		/// <returns></returns>
		virtual EventType GetEventType() const = 0;

		/// <summary>
		/// Gets the event's name.
		/// </summary>
		/// <returns></returns>
		virtual const char* GetName() const = 0;

		/// <summary>
		/// Gets event's category flags (ie. EventCategoryApplication or None).
		/// </summary>
		/// <returns></returns>
		virtual int GetCategoryFlags() const = 0;

		/// <summary>
		/// Converts the event into a string type.
		/// </summary>
		/// <returns></returns>
		virtual std::string ToString() const {
			return GetName();
		}

		/// <summary>
		/// Checks if the event is contained in the specified event category.
		/// </summary>
		/// <param name="category">Event category.</param>
		/// <returns>Whether the event is a member of the specified category.</returns>
		inline bool IsInCategory(EventCategory category)
		{
			return GetCategoryFlags() & category;
		}
	};

	class EventDispatcher
	{
		template<typename T>
		using EventFn = std::function<bool(T&)>;
	public:
		EventDispatcher(Event& event)
			: m_Event(event)
		{
		}

		/// <summary>
		/// Dispatches an event, if the callback function returns true the event will be marked as handled and will stop bubbling.
		/// </summary>
		/// <typeparam name="T">Event type.</typeparam>
		/// <param name="func">Event function.</param>
		/// <returns>Whether the event type is equal to that of the dispatch type.</returns>
		template<typename T>
		bool Dispatch(EventFn<T> func)
		{
			if (m_Event.GetEventType() == T::GetStaticType())
			{
				m_Event.Handled = func(*(T*)&m_Event);
				return true;
			}
			return false;
		}
	private:
		Event& m_Event;
	};

	inline std::ostream& operator<<(std::ostream& os, const Event& e)
	{
		return os << e.ToString();
	}
}

#endif // !EVENT_H_
