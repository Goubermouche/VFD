#ifndef APPLICATION_H_
#define APPLICATION_H_

#include  "FluidEngine/Platform/Windows/WindowsWindow.h"

namespace fe {
	class Application {
		public:
			Application();
			virtual ~Application();

			//void OnEvent(Event& e);
			void Run();
			void Close();

			static inline Application& Get() {
				return *s_Instance;
			}

			// get window

	private:
		/*bool OnWindowClose(WindowCloseEvent& e);
		bool OnWindowResize(WindowResizeEvent& e);*/

	private:
		std::unique_ptr<Window> m_Window;
		bool m_Running = true, m_Minimized = false;
		static Application* s_Instance;
	};
}

#endif // !APPLICATION_H_
