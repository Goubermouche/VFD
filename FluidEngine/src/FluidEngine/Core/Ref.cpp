#include "pch.h"
#include "Ref.h"


namespace fe {
	static std::unordered_set<void*> s_LiveReferences;
	static std::mutex s_LiveReferenceMutex;

	void RefUtils::AddToLiveReferences(void* instance)
	{
		std::scoped_lock<std::mutex> lock(s_LiveReferenceMutex);
		//HZ_CORE_ASSERT(instance);
		s_LiveReferences.insert(instance);
	}

	void RefUtils::RemoveFromLiveReferences(void* instance)
	{
		std::scoped_lock<std::mutex> lock(s_LiveReferenceMutex);
		//HZ_CORE_ASSERT(instance);
		//HZ_CORE_ASSERT(s_LiveReferences.find(instance) != s_LiveReferences.end());
		s_LiveReferences.erase(instance);
	}

	bool RefUtils::IsLive(void* instance)
	{
		//HZ_CORE_ASSERT(instance);
		return s_LiveReferences.find(instance) != s_LiveReferences.end();
	}
}

