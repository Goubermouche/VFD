#ifndef VIEWPORT_PANEL_H
#define VIEWPORT_PANEL_H

#include "Editor/Panels/EditorPanel.h"
#include "Renderer/Renderer.h"
#include "Editor/EditorCamera.h"

namespace vfd {
	class EditorCamera;

	class ViewportPanel : public EditorPanel
	{
	public:
		ViewportPanel();
		~ViewportPanel() override = default;

		void OnUpdate() override;
		void OnEvent(Event& event) override;
	private:
		bool OnSceneSaved(SceneSavedEvent& event);
		bool OnSceneLoaded(SceneLoadedEvent& event);
	private:
		Ref<FrameBuffer> m_FrameBuffer;
		Ref<EditorCamera> m_Camera;
		Ref<VertexArray> m_GridVAO;
		Ref<Material> m_GridMaterial;

		glm::vec2 m_Position;
		glm::vec2 m_Size;

		friend class EditorCamera;
	};
}

#endif // !VIEWPORT_PANEL_H