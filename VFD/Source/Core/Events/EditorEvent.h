#ifndef EDITOR_EVENT_H
#define EDITOR_EVENT_H

#include "Event.h"

namespace vfd {
	/// <summary>
	/// [Event] Called once before the active scene gets saved. 
	/// </summary>
	class SceneSavedEvent : public Event
	{
	public:
		SceneSavedEvent() = default;
		~SceneSavedEvent() override = default;

		EVENT_CLASS_TYPE(SceneSaved)
		EVENT_CLASS_CATEGORY(EventCategoryEditor)
	};

	/// <summary>
	/// [Event] Called once after a scene is loaded. 
	/// </summary>
	class SceneLoadedEvent : public Event
	{
	public:
		SceneLoadedEvent() = default;
		~SceneLoadedEvent() override = default;

		EVENT_CLASS_TYPE(SceneLoaded)
		EVENT_CLASS_CATEGORY(EventCategoryEditor)
	};

	class TimelineKeyUpdated : public Event
	{
	public:
		TimelineKeyUpdated(const unsigned int key)
			: m_Key(key)
		{}

		~TimelineKeyUpdated() override = default;

		unsigned int GetKey() const
		{
			return m_Key;
		}

		EVENT_CLASS_TYPE(TimelineKeyUpdated)
		EVENT_CLASS_CATEGORY(EventCategoryEditor)
	private:
		unsigned int m_Key;
	};
}

#endif // !EDITOR_EVENT_H
