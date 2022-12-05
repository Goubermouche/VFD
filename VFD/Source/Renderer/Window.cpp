#include "pch.h"
#include "Window.h"

#include <Glad/glad.h>

namespace vfd {
	static bool s_GLFWInitialized = false;

	Window::Window(const WindowDescription& desc)
	{
		m_Data.Title = desc.Title;
		m_Data.Width = desc.Width;
		m_Data.Height = desc.Height;

		if (s_GLFWInitialized == false) {
			const int initStatus = glfwInit();
			ASSERT(initStatus, "Failed to initialize GLFW!")
		    // glfwSetErrorCallback(GLFWErrorCallback);
			s_GLFWInitialized = true;
		}

		// glfwWindowHint(GLFW_SAMPLES, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		m_Window = glfwCreateWindow(desc.Width, desc.Height, desc.Title.c_str(), nullptr, nullptr);

		// Init context
		glfwMakeContextCurrent(m_Window);
		const int initStatus = gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
		ASSERT(initStatus, "Failed to initialize Glad!")
		// LOG("GLAD initialized successfully", "renderer", ConsoleColor::Purple);

		SetVSync(desc.VSync);

		glfwSetWindowUserPointer(m_Window, &m_Data);

#pragma region Callbacks
		glfwSetWindowSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
			{
				auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

				WindowResizeEvent event(static_cast<uint16_t>(width), static_cast<uint16_t>(height));
				data.EventCallback(event);
				data.Width = static_cast<uint16_t>(width);
				data.Height = static_cast<uint16_t>(height);
			});

		glfwSetWindowCloseCallback(m_Window, [](GLFWwindow* window)
			{
				const auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
				WindowCloseEvent event;
				data.EventCallback(event);
			});

		glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
			{
				const auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

				switch (action)
				{
				case GLFW_PRESS:
				{
					KeyPressedEvent event(static_cast<KeyCode>(key), 0);
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					KeyReleasedEvent event(static_cast<KeyCode>(key));
					data.EventCallback(event);
					break;
				}
				case GLFW_REPEAT:
				{
					KeyPressedEvent event(static_cast<KeyCode>(key), 1);
					data.EventCallback(event);
					break;
				}
				}
			});

		glfwSetCharCallback(m_Window, [](GLFWwindow* window, uint32_t codepoint)
			{
				const auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

				KeyTypedEvent event(static_cast<KeyCode>(codepoint));
				data.EventCallback(event);
			});

		glfwSetMouseButtonCallback(m_Window, [](GLFWwindow* window, int button, int action, int mods)
			{
				const auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

				switch (action)
				{
				case GLFW_PRESS:
				{
					MouseButtonPressedEvent event(static_cast<MouseButton>(button));
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					MouseButtonReleasedEvent event(static_cast<MouseButton>(button));
					data.EventCallback(event);
					break;
				}
				}
			});

		glfwSetScrollCallback(m_Window, [](GLFWwindow* window, double xOffset, double yOffset)
			{
				const auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
				MouseScrolledEvent event(static_cast<float>(xOffset), static_cast<float>(yOffset));
				data.EventCallback(event);
			});

		glfwSetCursorPosCallback(m_Window, [](GLFWwindow* window, double x, double y)
			{
				const auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
				MouseMovedEvent event(static_cast<float>(x), static_cast<float>(y));
				data.EventCallback(event);
			});

		glfwSetWindowIconifyCallback(m_Window, [](GLFWwindow* window, int iconified)
			{
				const auto& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
				WindowMinimizeEvent event(static_cast<bool>(iconified));
				data.EventCallback(event);
			});
#pragma endregion

		// LOG("window initialized successfully", "renderer][window", ConsoleColor::Purple);
	}

	void Window::ProcessEvents() 
	{
		glfwPollEvents();
	}

	void Window::SwapBuffers() const
	{
		glfwSwapBuffers(m_Window);
	}

	uint16_t Window::GetWidth() const
	{
		return m_Data.Width;
	}

	uint16_t Window::GetHeight() const
	{
		return m_Data.Height;
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

	void Window::SetEventCallback(const EventCallbackFn& callback)
	{
		m_Data.EventCallback = callback;
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

	void* Window::GetNativeWindow() const
	{
		return m_Window;
	};
}