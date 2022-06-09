#ifndef APPLICATION_H_
#define APPLICATION_H_

namespace fe {
	class Application {
		public:
			Application();
			virtual ~Application();

			//void OnEvent(Event& e);
			void Run();

			static inline Application& Get() {
				return *s_Instance;
			}

			// get window

	private:
		/*bool OnWindowClose(WindowCloseEvent& e);
		bool OnWindowResize(WindowResizeEvent& e);*/

	private:
		bool m_Running = true, m_Minimized = false;
		static Application* s_Instance;
	};
}

#endif // !APPLICATION_H_
