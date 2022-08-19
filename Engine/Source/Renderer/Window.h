#ifndef WINDOW_H
#define WINDOW_H

#include <GLFW/glfw3.h>

namespace fe {
	/// <summary>
	/// Window initializer description.
	/// </summary>
	struct WindowDesc {
		std::string Title;
		uint16_t Width;
		uint16_t Height;
		bool VSync;
	};

	/// <summary>
	/// Base window class
	/// </summary>
	class Window : public RefCounted {
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		Window(const WindowDesc& desc);
		~Window() = default;

		static void ProcessEvents();
		void SwapBuffers() const;

		uint16_t GetWidth() const {
			return m_Data.Width;
		}

		uint16_t GetHeight() const {
			return m_Data.Height;
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
			std::string Title;
			uint16_t Width = 0;
			uint16_t Height = 0;
			bool VSync = false;

			EventCallbackFn EventCallback;
		};

		WindowData m_Data;
	};
}

#endif // !WINDOW_H