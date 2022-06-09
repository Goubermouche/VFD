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

    Window* Window::Create(const WindowDesc& desc) {
        return new WindowsWindow(desc);
    }

    void WindowsWindow::OnUpdate()
    {
        glfwPollEvents();
        m_Context->SwapBuffers();
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
            //ASSERT(glfwInit(), "failed to initialize GLFW!");
            glfwInit();
            //glfwSetErrorCallback(GLFWErrorCallback);

            sGLFWInitialized = true;

            //SUCCESS("GLFW initialized", "window");
        }

        glfwWindowHint(GLFW_SAMPLES, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        m_Window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);

        m_Context = RendererContext::Create(m_Window);
        m_Context->Init();

        glfwSetWindowUserPointer(m_Window, &m_Data);
        SetVSync(true);
    }

    void WindowsWindow::Shutdown()
    {
    }
}
