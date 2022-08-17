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
		~EditorCamera() override = default;

		void OnEvent(Event& event);
	private:
		void UpdateProjection() override;
		void UpdateView() override;

		bool OnMouseScroll(MouseScrolledEvent& event);
		bool OnMouseMoved(MouseMovedEvent& event);

		void MousePan(const glm::vec2& delta);
		void MouseRotate(const glm::vec2& delta);
		void MouseZoom(float delta);

		glm::vec2 GetPanSpeed() const;
		float GetRotationSpeed() const;
		float GetZoomSpeed() const;
	private:
		glm::vec2 m_InitialMousePosition = { 0.0f, 0.0f };
		Ref<ViewportPanel> m_Context = nullptr;
	};
}

#endif // !EDITOR_CAMERA_H