#ifndef RENDERER_H_
#define RENDERER_H_

#include "RendererAPI.h"

namespace fe {
	class Renderer
	{
	public:
		static void Init();

		inline static RendererAPIType GetAPI() {
			return RendererAPI::GetAPI();
		}

		static void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

		static void SetClearColor(const glm::vec4& color);
		static void Clear();
	private:
		static RendererAPI* s_RendererAPI;
	};
}

#endif // !RENDERER_H_