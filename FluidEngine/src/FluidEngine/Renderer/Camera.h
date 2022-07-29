#ifndef CAMERA_H_
#define CAMERA_H_

namespace fe {
	/// <summary>
	/// Base camera class.
	/// </summary>
	class Camera : public RefCounted
	{
	public: 
		Camera(float fov, const glm::vec2& viewportSize, float nearClip, float farClip);

		void SetViewportSize(const glm::vec2& viewportSize);
		void SetPosition(const glm::vec3& position);

		glm::mat4& GetViewMatrix();
		glm::mat4& GetProjectionMatrix();
		glm::vec3& GetPosition();
		glm::quat GetOrientation() const;
		glm::vec2 GetViewportSize();
		float GetFOV();
	private:
		virtual void UpdateProjection() = 0;
		virtual void UpdateView() = 0;
	protected:
		glm::vec2 GetPanSpeed();
		float GetRotationSpeed();
		float GetZoomSpeed();

		glm::vec3 GetUpDirection() const;
		glm::vec3 GetRightDirection() const;
		glm::vec3 GetForwardDirection() const;

		glm::vec3 CalculatePosition();
	protected:
		glm::vec3 m_FocalPoint = { 0.0f, 0.0f, 0.0f };
		glm::vec3 m_Position;
		glm::vec2 m_ViewportSize;

		glm::mat4 m_ViewMatrix;
		glm::mat4 m_ProjectionMatrix = glm::mat4(1.0f);

		float m_FOV;
		float m_AspectRatio;
		float m_NearClip;
		float m_FarClip;
		float m_Distance = 15.0f;
		float m_Pitch = 0.0f;
		float m_Yaw = 0.0f;
	};
}

#endif // !CAMERA_H_



