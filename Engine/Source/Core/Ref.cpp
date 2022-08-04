#include "pch.h"
#include "Ref.h"

namespace fe {
	static std::unordered_set<void*> s_LiveReferences;
	static std::mutex s_LiveReferenceMutex;

	void AddToLiveReferences(void* instance)
	{
		std::scoped_lock<std::mutex> lock(s_LiveReferenceMutex);
		ASSERT(instance, "instance is out of scope!");
		s_LiveReferences.insert(instance);
	}

	void RemoveFromLiveReferences(void* instance)
	{
		std::scoped_lock<std::mutex> lock(s_LiveReferenceMutex);
		ASSERT(instance, "instance is out of scope!");
		ASSERT(s_LiveReferences.contains(instance), "instance is dead!");
		s_LiveReferences.erase(instance);
	}

	bool IsLive(void* instance)
	{
		ASSERT(instance, "instance is out of scope!");
		return s_LiveReferences.contains(instance);
	}
}