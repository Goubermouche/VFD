#ifndef RENDERER_CONTEXT_H_
#define RENDERER_CONTEXT_H_

#include "GLFW/glfw3.h"

namespace fe {
	/// <summary>
	/// Renderer Context for window objects. 
	/// </summary>
	class RendererContext : public RefCounted
	{
	public:
		RendererContext() = default;
		virtual ~RendererContext() = default;

		/// <summary>
		/// Initializes the context and the graphics library.
		/// </summary>
		virtual void Init() = 0;

		/// <summary>
		/// Swaps window buffers. 
		/// </summary>
		virtual void SwapBuffers() = 0;

		/// <summary>
		/// Creates a new Context based on the current RendererAPI.
		/// </summary>
		/// <param name="window">Native window pointer.</param>
		/// <returns>Reference to the newly created context.</returns>
		static Ref<RendererContext> Create(GLFWwindow* window);
	};
}

#endif // !RENDERER_CONTEXT_H_
