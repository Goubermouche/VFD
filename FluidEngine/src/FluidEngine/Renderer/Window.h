#ifndef WINDOW_H_
#define WINDOW_H_

namespace fe {
	/// <summary>
	/// Window initializer description.
	/// </summary>
	struct WindowDesc {
		std::string title;
		uint32_t width;
		uint32_t height;

		WindowDesc(const std::string& title = "Window",
			uint32_t width = 500, uint32_t height = 500)
			: title(title), width(width), height(height) {}
	};

	/// <summary>
	/// Base window class
	/// </summary>
	class Window {
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		virtual ~Window() {};
		virtual void ProcessEvents() = 0;
		virtual void SwapBuffers() = 0;

		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;

		/// <summary>
		/// Sets the window's event callback. If no event callback is set bad things will happen.
		/// </summary>
		/// <param name="callback">Event callback.</param>
		virtual void SetEventCallback(const EventCallbackFn& callback) = 0;
		virtual void SetTitle(const std::string& title) = 0;
		virtual void SetVSync(bool enabled) = 0;
		virtual bool IsVSync() const = 0;

		/// <summary>
		/// Gets the native window reference. 
		/// </summary>
		/// <returns>Window reference</returns>
		virtual void* GetNativeWindow() const = 0;

		/// <summary>
		/// Creates a new window.
		/// </summary>
		/// <param name="desc">Optional window desciption, the defaults can be found in Window.h</param>
		/// <returns>The newly created window.</returns>
		static Window* Create(const WindowDesc& desc = WindowDesc());
	};
}

#endif // !WINDOW_H_