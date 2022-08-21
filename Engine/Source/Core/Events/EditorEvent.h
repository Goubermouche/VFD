#ifndef EDITOR_EVENT_H
#define EDITOR_EVENT_H

#include "Event.h"

namespace fe {
	class SceneSavedEvent : public Event
	{
	public:
		SceneSavedEvent() = default;
		~SceneSavedEvent() override = default;

		EVENT_CLASS_TYPE(SceneSaved)
		EVENT_CLASS_CATEGORY(EventCategoryEditor)
	};

	class SceneLoadedEvent : public Event
	{
	public:
		SceneLoadedEvent() = default;
		~SceneLoadedEvent() override = default;

		EVENT_CLASS_TYPE(SceneLoaded)
		EVENT_CLASS_CATEGORY(EventCategoryEditor)
	};
}

#endif // !EDITOR_EVENT_H
