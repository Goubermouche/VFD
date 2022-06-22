#include "pch.h"
#include "RendererAPI.h"

namespace fe {
	RendererAPIType RendererAPI::s_API = RendererAPIType::None;

	void RendererAPI::SetAPI(RendererAPIType api)
	{
		s_API = api;
	}
}