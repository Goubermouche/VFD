#ifndef APPLICATION_H_
#define APPLICATION_H_

#include  "FluidEngine/Platform/Windows/WindowsWindow.h"

namespace fe {
	class Application {
		using EventCallbackFn = std::function<void(Event&)>;
	public:
			Application();
			virtual ~Application();

			void OnEvent(Event& event);

			void Run();
			void Close();

			//void AddEventCallback(const EventCallbackFn& eventCallback) { m_EventCallbacks.push_back(eventCallback); }

			template<typename Func>
			void QueueEvent(Func&& func)
			{
				m_EventQueue.push(func);
			}

			/// Creates & Dispatches an event either immediately, or adds it to an event queue which will be proccessed at the end of each frame
			template<typename TEvent, bool DispatchImmediately = false, typename... TEventArgs>
			void DispatchEvent(TEventArgs&&... args)
			{
				static_assert(std::is_assignable_v<Event, TEvent>);

				std::shared_ptr<TEvent> event = std::make_shared<TEvent>(std::forward<TEventArgs>(args)...);
				if constexpr (DispatchImmediately)
				{
					OnEvent(*event);
				}
				else
				{
					std::scoped_lock<std::mutex> lock(m_EventQueueMutex);
					m_EventQueue.push([event](){ Application::Get().OnEvent(*event); });
				}
			}

			static inline Application& Get() {
				return *s_Instance;
			}

			inline Window& GetWindow() {
				return *m_Window;
			}
	private:
		void ProcessEvents();

		bool OnWindowResize(WindowResizeEvent& e);
		bool OnWindowMinimize(WindowMinimizeEvent& e);
		bool OnWindowClose(WindowCloseEvent& e);
	private:
		std::unique_ptr<Window> m_Window;
		bool m_Running = true, m_Minimized = false;

		std::mutex m_EventQueueMutex;
		std::queue<std::function<void()>> m_EventQueue;
		//std::vector<EventCallbackFn> m_EventCallbacks;

		static Application* s_Instance;
	};
}

#endif // !APPLICATION_H_
