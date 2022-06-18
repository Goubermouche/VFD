#include "pch.h"
#include "EditorCamera.h"

#include "FluidEngine/Editor/Editor.h"

namespace fe {
	EditorCamera::EditorCamera(Ref<ViewportPanel> context, float fov, glm::vec2 viewportSize, float nearClip, float farClip)
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

		glm::quat orientation = GetOrientation();
		m_ViewMatrix = glm::translate(glm::mat4(1.0f), m_Position) * glm::toMat4(orientation);
		m_ViewMatrix = glm::inverse(m_ViewMatrix);
	}

	bool EditorCamera::OnMouseScroll(MouseScrolledEvent& e)
	{
		if (m_Context->m_Focused) {
			float delta = e.GetYOffset() * 0.1f;
			MouseZoom(delta);
			UpdateView();
		}
		return false;
	}

	bool EditorCamera::OnMouseMoved(MouseMovedEvent& e)
	{
		ImVec2 viewportPosition = m_Context->m_Position;

		const glm::vec2& mouse{ Input::GetMouseX() - viewportPosition.x, Input::GetMouseY() - viewportPosition.y };
		glm::vec2 delta = (mouse - m_InitialMousePosition) * 0.003f;

		m_InitialMousePosition = mouse;

		if (m_Context->m_Focused) {
			if (Editor::Get().GetCameraMode() ? Input::IsMouseButtonPressed(FE_MOUSE_BUTTON_LEFT) : Input::IsMouseButtonPressed(FE_MOUSE_BUTTON_MIDDLE)) {
				if (Input::IsKeyPressed(FE_KEY_LEFT_SHIFT)) {
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
		const float offset_amount = (glm::length(m_Position - m_FocalPoint) / m_Distance * m_Distance) / 2;

		const glm::vec3 x_axis = GetRightDirection() * -delta.x * offset_amount;
		const glm::vec3 y_axis = GetUpDirection() * -delta.y * offset_amount;

		m_FocalPoint = m_FocalPoint + x_axis - y_axis;
	}

	void EditorCamera::MouseRotate(const glm::vec2& delta)
	{
		float yawSign = GetUpDirection().y < 0 ? -1.0f : 1.0f;
		m_Yaw += yawSign * delta.x * GetRotationSpeed();
		m_Pitch += delta.y * GetRotationSpeed();
	}

	void EditorCamera::MouseZoom(float delta)
	{
		m_Distance -= delta * GetZoomSpeed();
		if (m_Distance < 1.0f)
		{
			m_FocalPoint += GetForwardDirection();
			m_Distance = 1.0f;
		}
	}
}