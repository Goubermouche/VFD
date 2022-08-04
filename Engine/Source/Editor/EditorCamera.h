#ifndef EDITOR_CAMERA_H
#define EDITOR_CAMERA_H

#include "Renderer/Camera.h"
#include "Editor/Panels/ViewportPanel.h"

namespace fe {
	class ViewportPanel;

	class EditorCamera : public Camera
	{
	public:
		EditorCamera(Ref<ViewportPanel> context, float fov, glm::vec2 viewportSize, float nearClip, float farClip);

		void OnEvent(Event& e);
	private:
		void UpdateProjection() override;
		void UpdateView() override;

		bool OnMouseScroll(MouseScrolledEvent& e);
		bool OnMouseMoved(MouseMovedEvent& e);

		void MousePan(const glm::vec2& delta);
		void MouseRotate(const glm::vec2& delta);
		void MouseZoom(float delta);
	private:
		glm::vec2 m_InitialMousePosition = { 0.0f, 0.0f };
		Ref<ViewportPanel> m_Context = nullptr;
	};
}

#endif // !EDITOR_CAMERA_H