#ifndef APPLICATION_H
#define APPLICATION_H

#include "Scene/AssetManager.h"
#include "Renderer/Window.h"
#include "Scene/Scene.h"
#include "Editor/Editor.h"

namespace vfd {
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
		void QueueEvent(Func&& func);
		
		/// <summary>
		/// Creates and dispatches an event either immediately, or adds it to the event queue which will be proccessed at the end of each frame.
		/// </summary>
		/// <typeparam name="TEvent">Event type.</typeparam>
		/// <typeparam name="...TEventArgs">Event arguments.</typeparam>
		/// <param name="...args">Event arguments.</param>
		template<typename TEvent, bool dispatchImmediately = false, typename... TEventArgs>
		void DispatchEvent(TEventArgs&&... args);
	
		/// <summary>
		/// Returns a reference to the application.
		/// </summary>
		/// <returns>Application reference.</returns>
		static Application& Get();

		/// <summary>
		/// Returns a reference to the main window.
		/// </summary>
		/// <returns>Window reference.</returns>
		Window& GetWindow();

		/// <summary>
		/// Returns the current scene context (the currently active scene). 
		/// </summary>
		/// <returns>A reference to the current scene context. </returns>
		Ref<Scene>& GetSceneContext();

		Ref<AssetManager>& GetAssetManager();
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
		Ref<Scene> m_SceneContext;
		Ref<AssetManager> m_AssetManager;

		bool m_Running = true;
		bool m_Minimized = false;

		// Events
		std::mutex m_EventQueueMutex;
		std::queue<std::function<void()>> m_EventQueue;

		/// <summary>
		/// Application instance singleton.
		/// </summary>
		static Application* s_Instance;
	};

	template<typename Func>
	inline void Application::QueueEvent(Func&& func)
	{
		m_EventQueue.push(func);
	}

	template<typename TEvent, bool dispatchImmediately, typename ...TEventArgs>
	inline void Application::DispatchEvent(TEventArgs && ...args)
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
			m_EventQueue.push([event](){ Get().OnEvent(*event); });
		}
	}
}

#endif // !APPLICATION_H
