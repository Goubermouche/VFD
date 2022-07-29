#ifndef WINDOW_H_
#define WINDOW_H_

#include <GLFW/glfw3.h>
#include <Glad/glad.h>

namespace fe {
	/// <summary>
	/// Window initializer description.
	/// </summary>
	struct WindowDesc {
		std::string title;
		uint32_t width;
		uint32_t height;
		bool VSync;

		WindowDesc(const std::string& title = "Window", bool vSync = false, 
			uint32_t width = 500, uint32_t height = 500)
			: title(title), VSync(vSync), width(width), height(height)
		{}
	};

	/// <summary>
	/// Base window class
	/// </summary>
	class Window : public RefCounted {
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		Window(const WindowDesc& desc);
		~Window() {};

		void ProcessEvents();
		void SwapBuffers();

		uint32_t GetWidth() const {
			return m_Data.width;
		}

		uint32_t GetHeight() const {
			return m_Data.height;
		}

		/// <summary>
		/// Sets the window's event callback. If no event callback is set bad things will happen.
		/// </summary>
		/// <param name="callback">Event callback.</param>
		void SetEventCallback(const EventCallbackFn& callback) {
			m_Data.EventCallback = callback;
		}

		void SetTitle(const std::string& title);
		void SetVSync(bool enabled);
		bool IsVSync() const;

		/// <summary>
		/// Gets the native window reference. 
		/// </summary>
		/// <returns>Window reference</returns>
		void* GetNativeWindow() const {
			return m_Window;
		};

	private:
		GLFWwindow* m_Window;

		struct WindowData {
			std::string title;
			uint32_t width;
			uint32_t height;
			bool VSync;

			EventCallbackFn EventCallback;
		};

		WindowData m_Data;
	};
}

#endif // !WINDOW_H_