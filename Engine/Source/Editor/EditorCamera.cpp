#include "pch.h"
#include "EditorCamera.h"

#include "Editor/Editor.h"

namespace fe {
	EditorCamera::EditorCamera(Ref<ViewportPanel> context, const float fov, const glm::vec2 viewportSize,const float nearClip,const float farClip, CameraType type)
		: Camera(fov, viewportSize, nearClip, farClip, type), m_Context(context)
	{
		UpdateView();
	}

	void EditorCamera::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<MouseScrolledEvent>(BIND_EVENT_FN(EditorCamera::OnMouseScroll));
		dispatcher.Dispatch<MouseMovedEvent>(BIND_EVENT_FN(EditorCamera::OnMouseMoved));
	}

	void EditorCamera::UpdateProjection()
	{
		// TODO: fix orthographic projection 
		m_AspectRatio = m_ViewportSize.x / m_ViewportSize.y;

		if (m_Type == CameraType::Orthographic) {
			m_ProjectionMatrix = glm::ortho(-m_AspectRatio, m_AspectRatio,	1.0f, 1.0f);
			ERR("not implemented properly (editor camera : projection)");
		}
		else {
			m_ProjectionMatrix = glm::perspective(glm::radians(m_FOV), m_AspectRatio, m_NearClip, m_FarClip);
		}
	}

	void EditorCamera::UpdateView()
	{
		m_Position = CalculatePosition();

		if (m_Type == CameraType::Orthographic) {
			m_ViewMatrix = glm::lookAt(m_Position, m_Pivot, GetUpDirection());
		}
		else {
			const glm::quat orientation = GetOrientation();
			m_ViewMatrix = glm::translate(glm::mat4(1.0f), m_Position) * glm::toMat4(orientation);
			m_ViewMatrix = glm::inverse(m_ViewMatrix);
		}
	}

	bool EditorCamera::OnMouseScroll(MouseScrolledEvent& event)
	{
		if (m_Context->m_Focused) {
			const float delta = event.GetYOffset() * 0.1f;
			MouseZoom(delta);
			UpdateView();
		}
		return false;
	}

	bool EditorCamera::OnMouseMoved(MouseMovedEvent& event)
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
		m_Pivot -= GetRightDirection() * delta.x * offsetAmmount.x * m_Distance;
		m_Pivot += GetUpDirection() * delta.y * offsetAmmount.y * m_Distance;
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

	glm::vec2 EditorCamera::GetPanSpeed() const
	{
		const float x = glm::min(m_ViewportSize.x / 1000, 2.4f); // max = 2.4f
		const float xFactor = 0.0666f * (x * x) - 0.2778f * x + 0.6021f;
		const float y = glm::min(m_ViewportSize.y / 1000, 2.4f); // max = 2.4f
		const float yFactor = 0.0666f * (y * y) - 0.2778f * y + 0.6021f;

		return { xFactor * 0.85f, yFactor * 0.85f };
	}

	float EditorCamera::GetRotationSpeed() const
	{
		return 1.8f;
	}

	float EditorCamera::GetZoomSpeed() const
	{
		float distance = m_Distance * 0.2f;
		distance = std::max(distance, 0.0f);
		float speed = distance * distance;
		speed = std::min(speed, 100.0f); // max speed = 100
		return speed; return 0.0f;
	}
}