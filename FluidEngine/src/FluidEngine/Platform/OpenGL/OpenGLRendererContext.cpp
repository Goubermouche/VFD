#include "pch.h"
#include "OpenGLRendererContext.h"

namespace fe::opengl {
	OpenGLRendererContext::OpenGLRendererContext(GLFWwindow* windowHandle)
		: m_WindowHandle(windowHandle)
	{
	}

	OpenGLRendererContext::~OpenGLRendererContext()
	{
	}

	void OpenGLRendererContext::Init()
	{
		glfwMakeContextCurrent(m_WindowHandle);
		//ASSERT(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress), "failed to initialize Glad!");
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	}

	void OpenGLRendererContext::SwapBuffers()
	{
		glfwSwapBuffers(m_WindowHandle);
	}
}
