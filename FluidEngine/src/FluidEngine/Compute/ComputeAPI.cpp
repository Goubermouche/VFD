#include "pch.h"
#include "ComputeAPI.h"

namespace fe {
	ComputeAPIType ComputeAPI::s_API = ComputeAPIType::None;
	DeviceInfo ComputeAPI::s_DeviceInfo;
	bool ComputeAPI::s_InitializedSuccessfully = false;

	void ComputeAPI::SetAPI(ComputeAPIType api)
	{
		s_API = api;
	}
}