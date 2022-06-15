#ifndef CAMERA_H_
#define CAMERA_H_

#include "FluidEngine/Editor/Panels/ViewportPanel.h"

namespace fe {
	class ViewportPanel;

	class EditorCamera : public RefCounted
	{
	public:
		EditorCamera(Ref<ViewportPanel> context, float fov, float aspectRatio, float nearClip, float farClip);

		void OnEvent(Event& e);

		void SetViewportSize(float width, float height);
		void SetPosition(const glm::vec3 position);

		glm::mat4& GetViewMatrix();
		glm::mat4& GetProjectionMatrix();
		glm::vec3& GetPosition();
		glm::quat GetOrientation() const;
		glm::vec2 GetSize();

		inline float GetFOV() {
			return m_FOV;
		}
	private:
		void UpdateProjection();
		void UpdateView();

		glm::vec3 CalculatePosition() const;

		bool OnMouseScroll(MouseScrolledEvent& e);
		bool OnMouseMoved(MouseMovedEvent& e);

		glm::vec2 GetPanSpeed() const;
		float GetRotationSpeed() const;
		float GetZoomSpeed() const;

		void MousePan(const glm::vec2& delta);
		void MouseRotate(const glm::vec2& delta);
		void MouseZoom(float delta);

		glm::vec3 GetUpDirection() const;
		glm::vec3 GetRightDirection() const;
		glm::vec3 GetForwardDirection() const;
	public:
		glm::vec3 m_FocalPoint = { 0.0f, 0.0f, 0.0f };
	private:
		float m_FOV;
		float m_AspectRatio;
		float m_NearClip;
		float m_FarClip;

		glm::mat4 m_ViewMatrix;
		glm::mat4 m_ProjMatrix = glm::mat4(1.0f);

		glm::vec3 m_Position;
		glm::vec2 m_InitialMousePosition = { 0.0f, 0.0f };

		float m_Distance = 10.0f;
		float m_Pitch = 0.0f;
		float m_Yaw = 0.0f;

		float m_ViewportWidth, m_ViewportHeight;

		Ref<ViewportPanel> m_Context;
	};
}

#endif // !CAMERA_H_
