#include "pch.h"
#include "Ref.h"


namespace fe {
	static std::unordered_set<void*> s_LiveReferences;
	static std::mutex s_LiveReferenceMutex;

	void RefUtils::AddToLiveReferences(void* instance)
	{
		std::scoped_lock<std::mutex> lock(s_LiveReferenceMutex);
		ASSERT(instance, "instance is out of scope!");
		s_LiveReferences.insert(instance);
	}

	void RefUtils::RemoveFromLiveReferences(void* instance)
	{
		std::scoped_lock<std::mutex> lock(s_LiveReferenceMutex);
		ASSERT(instance, "instance is out of scope!");
		ASSERT(s_LiveReferences.find(instance) != s_LiveReferences.end(), "instance is dead!");
		s_LiveReferences.erase(instance);
	}

	bool RefUtils::IsLive(void* instance)
	{
		ASSERT(instance, "instance is out of scope!");
		return s_LiveReferences.find(instance) != s_LiveReferences.end();
	}
}