#ifndef WINDOW_H_
#define WINDOW_H_

#include "pch.h"

namespace fe {
	struct WindowDesc {
		std::string title;
		uint32_t width;
		uint32_t height;

		WindowDesc(const std::string& title = "Window",
			uint32_t width = 500, uint32_t height = 500)
			: title(title), width(width), height(height){}
	};

	class Window {
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		virtual ~Window() {};
		virtual void ProcessEvents() = 0;
		virtual void SwapBuffers() = 0;

		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;

		virtual void SetEventCallback(const EventCallbackFn& callback) = 0;
		virtual void SetVSync(bool enabled) = 0;
		virtual void SetTitle(const std::string& title) = 0;
		virtual bool IsVSync() const = 0;

		virtual void* GetNativeWindow() const = 0;

		static Window* Create(const WindowDesc& desc = WindowDesc());
	};
}

#endif // !WINDOW_H_