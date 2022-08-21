#include "pch.h"
#include "Camera.h"

namespace fe {
	Camera::Camera(const float fov, const glm::vec2& viewportSize, const float nearClip, const float farClip, bool orthographic)
		:  m_ViewportSize(viewportSize), m_FOV(fov), m_AspectRatio(viewportSize.x / viewportSize.y),  m_NearClip(nearClip), m_FarClip(farClip), m_Orthographic(orthographic)
	{
	}

	void Camera::SetViewportSize(const glm::vec2& viewportSize)
	{
		m_ViewportSize = viewportSize;
		UpdateProjection();
	}

	void Camera::SetPosition(const glm::vec3& position)
	{
		const glm::vec3 d = position - m_Pivot;

		m_Pitch = std::atan2(d.y, std::sqrt(d.x * d.x + d.z * d.z));
		m_Yaw = std::atan2(d.z, d.x) - 1.5708f;
		m_Distance = glm::distance(m_Pivot, position);

		UpdateView();
	}

	void Camera::SetPivot(const glm::vec3& pivot)
	{
		m_Pivot = pivot;
		SetPosition(m_Position);
	}

	glm::mat4& Camera::GetViewMatrix()
	{
		return m_ViewMatrix;
	}

	glm::mat4& Camera::GetProjectionMatrix()
	{
		return m_ProjectionMatrix;
	}

	glm::vec3& Camera::GetPosition()
	{
		return m_Position;
	}

	glm::quat Camera::GetOrientation() const 
	{
		return glm::quat(glm::vec3(-m_Pitch, -m_Yaw, 0.0f));
	}

	glm::vec2 Camera::GetViewportSize() const
	{
		return m_ViewportSize;
	}

	const glm::vec3& Camera::GetPivot()
	{
		return m_Pivot;
	}

	float Camera::GetFOV() const
	{
		return m_FOV;
	}

	float Camera::GetNearClip() const
	{
		return m_NearClip;
	}

	float Camera::GetFarClip() const
	{
		return m_FarClip;
	}

	glm::vec3 Camera::CalculatePosition() const
	{
		return m_Pivot - GetForwardDirection() * m_Distance;
	}

	glm::vec3 Camera::GetUpDirection() const
	{
		return glm::rotate(GetOrientation(), glm::vec3(0.0f, 1.0f, 0.0f));
	}

	glm::vec3 Camera::GetRightDirection() const
	{
		return glm::rotate(GetOrientation(), glm::vec3(1.0f, 0.0f, 0.0f));
	}

	glm::vec3 Camera::GetForwardDirection() const
	{
		return glm::rotate(GetOrientation(), glm::vec3(0.0f, 0.0f, -1.0f));
	}
}