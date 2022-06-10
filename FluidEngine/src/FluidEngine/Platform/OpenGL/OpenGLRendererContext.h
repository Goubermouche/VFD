#ifndef OPENGL_RENDERER_CONTEXT_H_
#define OPENGL_RENDERER_CONTEXT_H_

#include "FluidEngine/Renderer/RendererContext.h"

namespace fe::opengl {
	class OpenGLRendererContext : public RendererContext
	{
	public:
		OpenGLRendererContext(GLFWwindow* windowHandle);
		virtual ~OpenGLRendererContext();

		virtual void Init() override;

		virtual void SwapBuffers() override;
	private:
		GLFWwindow* m_WindowHandle;
	};
}

#endif // !OPENGL_RENDERER_CONTEXT_H_

