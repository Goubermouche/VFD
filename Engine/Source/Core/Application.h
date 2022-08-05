#ifndef APPLICATION_H
#define APPLICATION_H

#include "Renderer/Window.h"
#include "Editor/Editor.h"

// Scene
#include "Scene/Scene.h"

namespace fe {
	/// <summary>
	/// Application singleton, manages the main process, and routes events.
	/// </summary>
	class Application final {
		using EventCallbackFunction = std::function<void(Event&)>;
	public:
		Application();
		~Application() = default;

		/// <summary>
		/// Called every time an event is dispatched with dispatchImmediately set to true (includes window events).
		/// </summary>
		/// <param name="event">Incoming event.</param>
		void OnEvent(Event& event);

		/// <summary>
		/// Main loop
		/// </summary>
		void Run();
		void Close();

		/// <summary>
		/// Enqueues a new event.
		/// </summary>
		/// <typeparam name="Func">Incoming eEvent function.</typeparam>
		/// <param name="func">Incoming Event function.</param>
		template<typename Func>
		void QueueEvent(Func&& func)
		{
			m_EventQueue.push(func);
		}

		/// <summary>
		/// Creates and dispatches an event either immediately, or adds it to an event queue which will be proccessed at the end of each frame.
		/// </summary>
		/// <typeparam name="TEvent">Event type.</typeparam>
		/// <typeparam name="...TEventArgs">Event arguments.</typeparam>
		/// <param name="...args">Event arguments.</param>
		template<typename TEvent, bool dispatchImmediately = false, typename... TEventArgs>
		void DispatchEvent(TEventArgs&&... args)
		{
			static_assert(std::is_assignable_v<Event, TEvent>);

			std::shared_ptr<TEvent> event = std::make_shared<TEvent>(std::forward<TEventArgs>(args)...);
			if constexpr (dispatchImmediately)
			{
				OnEvent(*event);
			}
			else
			{
				std::scoped_lock<std::mutex> lock(m_EventQueueMutex);
				m_EventQueue.push([event](){ Application::Get().OnEvent(*event); });
			}
		}

		/// <summary>
		/// Gets a reference to the application.
		/// </summary>
		/// <returns>Application reference.</returns>
		static Application& Get() {
			return *s_Instance;
		}

		/// <summary>
		/// Gets a reference to the main window.
		/// </summary>
		/// <returns>Window reference.</returns>
		Window& GetWindow() {
			return *m_Window;
		}

		// Core API for saving & loading scenes
		void SaveCurrentSceneContext(const std::string& filepath);
		void LoadSceneContext(const std::string& filepath);
	private:
		/// <summary>
		/// Processes events that do not require immediate execution.
		/// </summary>
		void ProcessEvents();

		// Application events
		bool OnWindowResize(WindowResizeEvent& e);
		bool OnWindowMinimize(WindowMinimizeEvent& e);
		bool OnWindowClose(WindowCloseEvent& e);
	private:
		Ref<Window> m_Window;
		Ref<Editor> m_Editor;

		bool m_Running = true;
		bool m_Minimized = false;

		// Events
		std::mutex m_EventQueueMutex;
		std::queue<std::function<void()>> m_EventQueue;

		// Scene
		Ref<Scene> m_SceneContext;

		/// <summary>
		/// Application instance singleton.
		/// </summary>
		static Application* s_Instance;
	};
}

#endif // !APPLICATION_H
