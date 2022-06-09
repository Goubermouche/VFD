#include "pch.h"
#include "OpenGLRendererContext.h"

fe::OpenGLRendererContext::OpenGLRendererContext(GLFWwindow* windowHandle)
	: m_WindowHandle(windowHandle)
{
}

fe::OpenGLRendererContext::~OpenGLRendererContext()
{
}

void fe::OpenGLRendererContext::Init()
{
	glfwMakeContextCurrent(m_WindowHandle);
	//ASSERT(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress), "failed to initialize Glad!");
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
}

void fe::OpenGLRendererContext::SwapBuffers()
{
	glfwSwapBuffers(m_WindowHandle);
}
