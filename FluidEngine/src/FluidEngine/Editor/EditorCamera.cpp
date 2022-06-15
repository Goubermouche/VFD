#include "pch.h"
#include "EditorCamera.h"

namespace fe {
	EditorCamera::EditorCamera(Ref<ViewportPanel> context, float fov, float aspectRatio, float nearClip, float farClip)
		:m_Context(context), m_FOV(fov), m_AspectRatio(aspectRatio), m_NearClip(nearClip), m_FarClip(farClip), m_ProjMatrix(glm::perspective(glm::radians(fov), aspectRatio, nearClip, farClip))
	{
		UpdateView();
	}

	void EditorCamera::OnEvent(Event& e)
	{
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<MouseScrolledEvent>(BIND_EVENT_FN(EditorCamera::OnMouseScroll));
		dispatcher.Dispatch<MouseMovedEvent>(BIND_EVENT_FN(EditorCamera::OnMouseMoved));
	}

	void EditorCamera::SetViewportSize(float width, float height)
	{
		m_ViewportWidth = width;
		m_ViewportHeight = height;
		UpdateProjection();
	}

	void EditorCamera::SetPosition(const glm::vec3 position)
	{
		ASSERT(false, "not implemented!");
	}

	glm::mat4& EditorCamera::GetViewMatrix()
	{
		return m_ViewMatrix;
	}

	glm::mat4& EditorCamera::GetProjectionMatrix()
	{
		return m_ProjMatrix;
	}

	glm::vec3& EditorCamera::GetPosition()
	{
		return m_Position;
	}

	glm::quat EditorCamera::GetOrientation() const
	{
		return glm::quat(glm::vec3(-m_Pitch, -m_Yaw, 0.0f));
	}

	glm::vec2 EditorCamera::GetSize()
	{
		return { m_ViewportWidth, m_ViewportHeight };
	}

	void EditorCamera::UpdateProjection()
	{
		m_AspectRatio = m_ViewportWidth / m_ViewportHeight;
		m_ProjMatrix = glm::perspective(glm::radians(m_FOV), m_AspectRatio, m_NearClip, m_FarClip);
	}

	void EditorCamera::UpdateView()
	{
		m_Position = CalculatePosition();

		glm::quat orientation = GetOrientation();
		m_ViewMatrix = glm::translate(glm::mat4(1.0f), m_Position) * glm::toMat4(orientation);
		m_ViewMatrix = glm::inverse(m_ViewMatrix);
	}

	glm::vec3 EditorCamera::CalculatePosition() const
	{
		return m_FocalPoint - GetForwardDirection() * m_Distance;
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
			if (false ? Input::IsMouseButtonPressed(FE_MOUSE_BUTTON_LEFT) : Input::IsMouseButtonPressed(FE_MOUSE_BUTTON_MIDDLE)) {
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

	glm::vec2 EditorCamera::GetPanSpeed() const
	{
		float x = std::min(m_ViewportWidth / 1000.0f, 5.4f);
		float xFactor = 0.0666f * (x * x) - 0.1778f * x + 0.3021f;

		float y = std::min(m_ViewportHeight / 1000.0f, 5.4f);
		float yFactor = 0.0666f * (y * y) - 0.1778f * y + 0.3021f;

		return { xFactor, yFactor };
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
		return speed;
	}

	void EditorCamera::MousePan(const glm::vec2& delta)
	{
		const float offset_amount = (glm::length(m_Position - m_FocalPoint) / m_Distance * m_Distance) / 2;

		const glm::vec3 x_axis = GetRightDirection() * -delta.x * offset_amount;
		const glm::vec3 y_axis = GetUpDirection() * -delta.y * offset_amount;

		m_FocalPoint = m_FocalPoint + x_axis - y_axis;

		/*glm::vec2 speed = GetPanSpeed();
		m_FocalPoint += -GetRightDirection() * delta.x * speed.x * m_Distance;
		m_FocalPoint += GetUpDirection() * delta.y * speed.y * m_Distance;*/
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

	glm::vec3 EditorCamera::GetUpDirection() const
	{
		return glm::rotate(GetOrientation(), glm::vec3(0.0f, 1.0f, 0.0f));
	}

	glm::vec3 EditorCamera::GetRightDirection() const
	{
		return glm::rotate(GetOrientation(), glm::vec3(1.0f, 0.0f, 0.0f));
	}

	glm::vec3 EditorCamera::GetForwardDirection() const
	{
		return glm::rotate(GetOrientation(), glm::vec3(0.0f, 0.0f, -1.0f));
	}
}