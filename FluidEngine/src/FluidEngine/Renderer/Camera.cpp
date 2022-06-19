#include "pch.h"
#include "Camera.h"

namespace fe {
	Camera::Camera(float fov, const glm::vec2& viewportSize, float nearClip, float farClip)
		: m_FOV(fov), m_AspectRatio(viewportSize.x / viewportSize.y), m_ViewportSize(viewportSize), m_NearClip(nearClip), m_FarClip(farClip)
	{
	}

	void Camera::SetViewportSize(const glm::vec2& viewportSize)
	{
		m_ViewportSize = viewportSize;
		UpdateProjection();
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

	glm::vec2 Camera::GetViewportSize()
	{
		return m_ViewportSize;
	}

	float Camera::GetFOV()
	{
		return m_FOV;
	}

	glm::vec3 Camera::CalculatePosition()
	{
		return m_FocalPoint - GetForwardDirection() * m_Distance;
	}

	glm::vec2 Camera::GetPanSpeed()
	{
		float x = std::min(m_ViewportSize.x / 1000.0f, 5.4f);
		float xFactor = 0.0666f * (x * x) - 0.1778f * x + 0.3021f;
		float y = std::min(m_ViewportSize.y / 1000.0f, 5.4f);
		float yFactor = 0.0666f * (y * y) - 0.1778f * y + 0.3021f;
		return { xFactor, yFactor };
	}

	float Camera::GetRotationSpeed()
	{
		return 1.8f;
	}

	float Camera::GetZoomSpeed()
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