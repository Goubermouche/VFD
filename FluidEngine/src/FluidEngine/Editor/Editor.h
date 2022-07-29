#ifndef EDITOR_H_
#define EDITOR_H_

#include "Panels/PanelManager.h"

namespace fe {
	class Editor : public RefCounted {
	public:
		Editor();
		~Editor();

		void OnUpdate();
		void OnEvent(Event& event);

		/// <summary>
		/// Sets the scene context for the editor and all child panels.
		/// </summary>
		/// <param name="context">New scene context.</param>
		void SetSceneContext(Ref<Scene> context);
		
		/// <summary>
		/// Sets the selection context for the editor and all child panels. This will eventually support multiple selected items.
		/// </summary>
		/// <param name="selectionContext">The currently selected entity.</param>
		void SetSelectionContext(Entity context);

		// Save & Load functions that open their respective file dialog and call the Application's save & load API.
		void SaveCurrentSceneContext();
		void LoadSceneContext();

		bool GetCameraMode(); // Temp

		static inline Editor& Get() {
			return *s_Instance;
		}
	private:
		void InitImGui();

		// Events
		bool OnKeyPressed(KeyPressedEvent& e);
	private:
		std::unique_ptr<PanelManager> m_PanelManager;
		Ref<Scene> m_SceneContext;
		Entity m_SelectionContext;

		// Temp utility 
		bool m_StyleEditorEnabled = false;
		bool m_ImGuiDemoWindowEnabled = false;
		bool m_CameraTrackpadMode = false;

		static Editor* s_Instance;
	};
}

#endif // !EDITOR_H_