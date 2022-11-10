#ifndef EDITOR_H
#define EDITOR_H

#include "Editor/Panels/PanelManager.h"
#include "Editor/Panels/ReadMePanel.h"
#include "Scene/AssetManager.h"

namespace fe {
	enum class CameraControlMode
	{
		None = 0,
		Mouse,
		TrackPad
	};

	class Editor : public RefCounted {
	public:
		Editor();
		~Editor() = default;

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
		/// <param name="context">The currently selected entity.</param>
		void SetSelectionContext(Entity context);

		Entity GetSelectionContext() {
			return m_SelectionContext;
		}

		// Save & Load functions that open their respective file dialog and call the Application's save & load API.
		void SaveCurrentSceneContext();

		/// <summary>
		/// Opens a file dialog window and lets the user choose the target scene. 
		/// </summary>
		void LoadSceneContext();

		CameraControlMode GetCameraMode() const; // TEMP

		static Editor& Get() {
			return *s_Instance;
		}

		Ref<AssetManager>& GetAssetManager() {
			return m_AssetManager;
		}
	private:
		// Events
		bool OnKeyPressed(KeyPressedEvent& event);

		bool OnSceneSaved(SceneSavedEvent& event);
		bool OnSceneLoaded(SceneLoadedEvent& event);
	private:
		Ref<Scene> m_SceneContext;
		Ref<AssetManager> m_AssetManager; // registry for assets used only in the editor
		Ref<PanelManager> m_PanelManager;

		Ref<ReadMePanel> m_ReadMePanel;

		Entity m_SelectionContext;

		// Temp utility 
		CameraControlMode m_CameraControlMode = CameraControlMode::Mouse;

		static Editor* s_Instance;
	};
}

#endif // !EDITOR_H