#ifndef EDITOR_CAMERA_H
#define EDITOR_CAMERA_H

#include "Renderer/Camera.h"
#include "Editor/Panels/ViewportPanel.h"

namespace vfd {
	class ViewportPanel;

	class EditorCamera : public Camera
	{
	public:
		EditorCamera(Ref<ViewportPanel> context, float fov, glm::vec2 viewportSize, float nearClip, float farClip, float rotationSpeed, CameraType type = CameraType::Perspective);
		~EditorCamera() override = default;

		void OnEvent(Event& event);
	private:
		void UpdateProjection() override;
		void UpdateView() override;
		void MousePan(const glm::vec2& delta);
		void MouseRotate(const glm::vec2& delta);
		void MouseZoom(float delta);

		bool OnMouseScroll(MouseScrolledEvent& event);
		bool OnMouseMoved(MouseMovedEvent& event);

		float GetZoomSpeed() const;

		glm::vec2 GetPanSpeed() const;
	private:
		Ref<ViewportPanel> m_Context;
		glm::vec2 m_InitialMousePosition = { 0.0f, 0.0f };
		float m_RotationSpeed;
	};
}

#endif // !EDITOR_CAMERA_H