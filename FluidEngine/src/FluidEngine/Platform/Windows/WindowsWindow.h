#ifndef WINDOWS_WINDOW_H_
#define WINDOWS_WINDOW_H_

#include "FluidEngine/Renderer/Window.h"
;
#include <GLFW/glfw3.h>
#include <Glad/glad.h>

namespace fe {
	class WindowsWindow : public Window {
	public:
		WindowsWindow(const WindowDesc& desc);
		virtual ~WindowsWindow();

		virtual void ProcessEvents() override;
		virtual void SwapBuffers() override;

		inline uint32_t GetWidth() const override {
			return m_Data.width;
		}
		inline uint32_t GetHeight() const override {
			return m_Data.height;
		}

		virtual void SetEventCallback(const EventCallbackFn& callback) override {
			m_Data.EventCallback = callback;
		}

		virtual void SetVSync(bool enabled) override;
		virtual void SetTitle(const std::string& title) override;
		virtual bool IsVSync() const override;

		virtual void* GetNativeWindow() const {
			return m_Window;
		};

	private:
		virtual void Init(const WindowDesc& desc);
		virtual void Shutdown();
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

#endif // !WINDOWS_WINDOW_H_