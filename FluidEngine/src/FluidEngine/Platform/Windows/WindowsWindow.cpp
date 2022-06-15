#include "pch.h"
#include "WindowsWindow.h"


namespace fe {
	static bool sGLFWInitialized = false;

	WindowsWindow::WindowsWindow(const WindowDesc& props)
	{
		Init(props);
	}

	WindowsWindow::~WindowsWindow()
	{

	}

	void WindowsWindow::ProcessEvents()
	{
		glfwPollEvents();
		//Input::Update();
	}

	void WindowsWindow::SwapBuffers()
	{
		m_RendererContext->SwapBuffers();
	}

	Window* Window::Create(const WindowDesc& desc) {
		return new WindowsWindow(desc);
	}

	void WindowsWindow::SetVSync(bool enabled)
	{
		if (enabled) {
			glfwSwapInterval(1);
		}
		else {
			glfwSwapInterval(0);
		}

		m_Data.VSync = enabled;
	}

	void WindowsWindow::SetTitle(const std::string& title)
	{
		glfwSetWindowTitle(m_Window, title.c_str());
		m_Data.title = title;
	}

	bool WindowsWindow::IsVSync() const
	{
		return m_Data.VSync;
	}

	void WindowsWindow::Init(const WindowDesc& desc)
	{
		m_Data.title = desc.title;
		m_Data.width = desc.width;
		m_Data.height = desc.height;

		if (sGLFWInitialized == false) {
			ASSERT(glfwInit(), "failed to initialize GLFW!");
			glfwInit();
			//glfwSetErrorCallback(GLFWErrorCallback);

			sGLFWInitialized = true;

			LOG("GLFW initialized successfully");
		}

		glfwWindowHint(GLFW_SAMPLES, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		m_Window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);

		m_RendererContext = RendererContext::Create(m_Window);
		m_RendererContext->Init();

		glfwSetWindowUserPointer(m_Window, &m_Data);
		SetVSync(true);

#pragma region Callbacks
		glfwSetWindowSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
			{
				auto& data = *((WindowData*)glfwGetWindowUserPointer(window));

				WindowResizeEvent event((uint32_t)width, (uint32_t)height);
				data.EventCallback(event);
				data.width = width;
				data.height = height;
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

		LOG("window initialized successfully");
	}

	void WindowsWindow::Shutdown()
	{
	}
}
