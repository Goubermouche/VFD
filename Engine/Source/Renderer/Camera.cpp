#include "pch.h"
#include "Camera.h"

namespace fe {
	Camera::Camera(const float fov, const glm::vec2& viewportSize, const float nearClip, const float farClip)
		:  m_ViewportSize(viewportSize), m_FOV(fov), m_AspectRatio(viewportSize.x / viewportSize.y),  m_NearClip(nearClip), m_FarClip(farClip)
	{
	}

	void Camera::SetViewportSize(const glm::vec2& viewportSize)
	{
		m_ViewportSize = viewportSize;
		UpdateProjection();
	}

	void Camera::SetPosition(const glm::vec3& position)
	{
		const glm::vec3 d = position - m_FocalPoint;

		m_Pitch = std::atan2(d.y, std::sqrt(d.x * d.x + d.z * d.z));
		m_Yaw = std::atan2(d.z, d.x) - 1.5708f;

		m_Distance = glm::distance(m_FocalPoint, position);

		UpdateView();
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
		return m_FocalPoint - GetForwardDirection() * m_Distance;
	}

	glm::vec2 Camera::GetPanSpeed() const
	{
		const float x = glm::min(m_ViewportSize.x / 1000, 2.4f); // max = 2.4f
		const float xFactor = 0.0666f * (x * x) - 0.2778f * x + 0.6021f;
		const float y = glm::min(m_ViewportSize.y / 1000, 2.4f); // max = 2.4f
		const float yFactor = 0.0666f * (y * y) - 0.2778f * y + 0.6021f;

		return { xFactor * 0.85f, yFactor * 0.85f };
	}

	float Camera::GetRotationSpeed() const 
	{
		return 1.8f;
	}

	float Camera::GetZoomSpeed() const
	{
		float distance = m_Distance * 0.2f;
		distance = std::max(distance, 0.0f);
		float speed = distance * distance;
		speed = std::min(speed, 100.0f); // max speed = 100
		return speed;
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