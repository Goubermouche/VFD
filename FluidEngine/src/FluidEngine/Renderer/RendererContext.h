#ifndef RENDERER_CONTEXT_H_
#define RENDERER_CONTEXT_H_

namespace fe {
	class RendererContext
	{
	public:
		RendererContext() = default;
		virtual ~RendererContext() = default;

		virtual void Init() = 0;
		virtual void SwapBuffers() = 0;

		static std::shared_ptr<RendererContext> Create(GLFWwindow* window);
	};
}

#endif // !RENDERER_CONTEXT_H_
