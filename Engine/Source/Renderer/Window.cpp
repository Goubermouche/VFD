#include "pch.h"
#include "Window.h"

#include <Glad/glad.h>

namespace fe {
	static bool s_GLFWInitialized = false;

	Window::Window(const WindowDesc& desc)
	{
		m_Data.Title = desc.Title;
		m_Data.Width = desc.Width;
		m_Data.Height = desc.Height;

		if (s_GLFWInitialized == false) {
			ASSERT(glfwInit(), "failed to initialize GLFW!");
			glfwInit();
		    // glfwSetErrorCallback(GLFWErrorCallback);
			s_GLFWInitialized = true;

			LOG("GLFW initialized successfully", "renderer][window", ConsoleColor::Purple);
		}

		// glfwWindowHint(GLFW_SAMPLES, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		m_Window = glfwCreateWindow(desc.Width, desc.Height, desc.Title.c_str(), nullptr, nullptr);

		// Init context
		glfwMakeContextCurrent(m_Window);
		ASSERT(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress), "failed to initialize Glad!");
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		LOG("GLAD initialized successfully", "renderer", ConsoleColor::Purple);

		SetVSync(desc.VSync);

		glfwSetWindowUserPointer(m_Window, &m_Data);

#pragma region Callbacks
		glfwSetWindowSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));

				WindowResizeEvent event((uint32_t)width, (uint32_t)height);
				data.EventCallback(event);
				data.Width = width;
				data.Height = height;
			});

		glfwSetWindowCloseCallback(m_Window, [](GLFWwindow* window)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));

				WindowCloseEvent event;
				data.EventCallback(event);
			});

		glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));

				switch (action)
				{
				case GLFW_PRESS:
				{
					KeyPressedEvent event((KeyCode)key, 0);
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					KeyReleasedEvent event((KeyCode)key);
					data.EventCallback(event);
					break;
				}
				case GLFW_REPEAT:
				{
					KeyPressedEvent event((KeyCode)key, 1);
					data.EventCallback(event);
					break;
				}
				}
			});

		glfwSetCharCallback(m_Window, [](GLFWwindow* window, uint32_t codepoint)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));

				KeyTypedEvent event((KeyCode)codepoint);
				data.EventCallback(event);
			});

		glfwSetMouseButtonCallback(m_Window, [](GLFWwindow* window, int button, int action, int mods)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));

				switch (action)
				{
				case GLFW_PRESS:
				{
					MouseButtonPressedEvent event((MouseButton)button);
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					MouseButtonReleasedEvent event((MouseButton)button);
					data.EventCallback(event);
					break;
				}
				}
			});

		glfwSetScrollCallback(m_Window, [](GLFWwindow* window, double xOffset, double yOffset)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));

				MouseScrolledEvent event((float)xOffset, (float)yOffset);
				data.EventCallback(event);
			});

		glfwSetCursorPosCallback(m_Window, [](GLFWwindow* window, double x, double y)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));
				MouseMovedEvent event((float)x, (float)y);
				data.EventCallback(event);
			});

		glfwSetWindowIconifyCallback(m_Window, [](GLFWwindow* window, int iconified)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));
				WindowMinimizeEvent event((bool)iconified);
				data.EventCallback(event);
			});
#pragma endregion

		LOG("window initialized successfully", "renderer][window", ConsoleColor::Purple);
	}

	void Window::ProcessEvents() 
	{
		glfwPollEvents();
	}

	void Window::SwapBuffers() const
	{
		glfwSwapBuffers(m_Window);
	}

	void Window::SetVSync(bool enabled)
	{
		if (enabled) {
			glfwSwapInterval(1);
		}
		else {
			glfwSwapInterval(0);
		}

		m_Data.VSync = enabled;
	}

	void Window::SetTitle(const std::string& title)
	{
		glfwSetWindowTitle(m_Window, title.c_str());
		m_Data.Title = title;
	}

	bool Window::IsVSync() const
	{
		return m_Data.VSync;
	}
}