#ifndef RENDERER_API_H_
#define RENDERER_API_H_

namespace fe {

	enum class RendererAPIType
	{
		None,
		OpenGL
	};

	class RendererAPI
	{
	public:
		virtual void Init() = 0;

		virtual void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) = 0;

		virtual void SetClearColor(const glm::vec4& color) = 0;
		virtual void Clear() = 0;

		static inline RendererAPIType GetAPI() {
			return s_API; 
		}

		static void SetAPI(RendererAPIType api);
	private:
		static RendererAPIType s_API;
	};
}

#endif // !RENDERER_API_H_