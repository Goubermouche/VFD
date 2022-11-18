#ifndef EVENT_H
#define EVENT_H

#include "pch.h"

namespace vfd {
	enum class EventType
	{
		None = 0,
		WindowClose, 
		WindowMinimize,
		WindowResize,
		WindowFocus,
		WindowLostFocus,
		WindowMoved,
		WindowTitleBarHitTest,
		KeyPressed,
		KeyReleased, 
		KeyTyped,
		MouseButtonPressed,
		MouseButtonReleased,
		MouseMoved, 
		MouseScrolled,

		SceneSaved, 
		SceneLoaded
	};

	enum EventCategory
	{
		None = 0,
		EventCategoryApplication = 1 << 0,
		EventCategoryInput = 1 << 1,
		EventCategoryKeyboard = 1 << 2,
		EventCategoryMouse = 1 << 3,
		EventCategoryMouseButton = 1 << 4, 
		EventCategoryEditor = 1 << 5
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
		virtual ~Event() = default;

		/// <summary>
		/// Gets the event type.
		/// </summary>
		/// <returns></returns>
		[[nodiscard]]
		virtual EventType GetEventType() const = 0;

		/// <summary>
		/// Gets the event's name.
		/// </summary>
		/// <returns></returns>
		[[nodiscard]]
		virtual const char* GetName() const = 0;

		/// <summary>
		/// Gets event's category flags (ie. EventCategoryApplication or None).
		/// </summary>
		/// <returns></returns>
		[[nodiscard]]
		virtual int GetCategoryFlags() const = 0;

		/// <summary>
		/// Converts the event into a string type.
		/// </summary>
		/// <returns></returns>
		[[nodiscard]]
		virtual const std::string& ToString() const {
			return GetName();
		}

		/// <summary>
		/// Checks if the event is contained in the specified event category.
		/// </summary>
		/// <param name="category">Event category.</param>
		/// <returns>Whether the event is a member of the specified category.</returns>
		[[nodiscard]]
		bool IsInCategory(const EventCategory category) const
		{
			return GetCategoryFlags() & category;
		}
	public:
		bool handled = false;
	};

	class EventDispatcher
	{
		template<typename T>
		using EventFunction = std::function<bool(T&)>;
	public:
		EventDispatcher(Event& event)
			: m_Event(event)
		{}

		~EventDispatcher() = default;

		/// <summary>
		/// Dispatches an event, if the callback function returns true the event will be marked as handled and will stop bubbling.
		/// </summary>
		/// <typeparam name="T">Event type.</typeparam>
		/// <param name="func">Event function.</param>
		/// <returns>Whether the event type is equal to that of the dispatch type.</returns>
		template<typename T>
		bool Dispatch(EventFunction<T> func)
		{
			if (m_Event.GetEventType() == T::GetStaticType())
			{
				m_Event.handled = func(*(T*)&m_Event);
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

#endif // !EVENT_H
