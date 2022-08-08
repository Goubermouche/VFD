#include "pch.h"
#include "EditorCamera.h"

#include "Editor/Editor.h"

namespace fe {
	EditorCamera::EditorCamera(Ref<ViewportPanel> context, const float fov, const glm::vec2 viewportSize,const float nearClip,const float farClip)
		: Camera(fov, viewportSize, nearClip, farClip), m_Context(context)
	{
		UpdateView();
	}

	void EditorCamera::OnEvent(Event& e)
	{
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<MouseScrolledEvent>(BIND_EVENT_FN(EditorCamera::OnMouseScroll));
		dispatcher.Dispatch<MouseMovedEvent>(BIND_EVENT_FN(EditorCamera::OnMouseMoved));
	}

	void EditorCamera::UpdateProjection()
	{
		m_AspectRatio = m_ViewportSize.x / m_ViewportSize.y;
		m_ProjectionMatrix = glm::perspective(glm::radians(m_FOV), m_AspectRatio, m_NearClip, m_FarClip);
	}

	void EditorCamera::UpdateView()
	{
		m_Position = CalculatePosition();

		const glm::quat orientation = GetOrientation();
		m_ViewMatrix = glm::translate(glm::mat4(1.0f), m_Position) * glm::toMat4(orientation);
		m_ViewMatrix = glm::inverse(m_ViewMatrix);
	}

	bool EditorCamera::OnMouseScroll(MouseScrolledEvent& e)
	{
		if (m_Context->m_Focused) {
			const float delta = e.GetYOffset() * 0.1f;
			MouseZoom(delta);
			UpdateView();
		}
		return false;
	}

	bool EditorCamera::OnMouseMoved(MouseMovedEvent& e)
	{
		const glm::vec2 viewportPosition = m_Context->m_Position;
		const glm::vec2& mouse{ Input::GetMouseX() - viewportPosition.x, Input::GetMouseY() - viewportPosition.y };
		const glm::vec2 delta = (mouse - m_InitialMousePosition) * 0.003f;
		m_InitialMousePosition = mouse;

		if (m_Context->m_Focused) {
			if (Editor::Get().GetCameraMode() == CameraControlMode::Mouse ? Input::IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE) : Input::IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
				if (Input::IsKeyPressed(KEY_LEFT_SHIFT)) {
					MousePan(delta);
				}
				else {
					MouseRotate(delta);
				}
			}

			UpdateView();
		}

		return false;
	}

	void EditorCamera::MousePan(const glm::vec2& delta)
	{
		glm::vec2 offsetAmmount = GetPanSpeed();
		m_FocalPoint -= GetRightDirection() * delta.x * offsetAmmount.x * m_Distance;
		m_FocalPoint += GetUpDirection() * delta.y * offsetAmmount.y * m_Distance;
	}

	void EditorCamera::MouseRotate(const glm::vec2& delta)
	{
		const float yawSign = GetUpDirection().y < 0 ? -1.0f : 1.0f;
		m_Yaw += yawSign * delta.x * GetRotationSpeed();
		m_Pitch += delta.y * GetRotationSpeed();
	}

	void EditorCamera::MouseZoom(const float delta)
	{
		if (m_Distance - delta * GetZoomSpeed() > 0.01f) {
			m_Distance -= delta * GetZoomSpeed();
		}
	}
}