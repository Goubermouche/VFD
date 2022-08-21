#ifndef CAMERA_H
#define CAMERA_H

namespace fe {
	enum class CameraType {
		None = 0,
		Perspective, 
		Orthographic
	};

	/// <summary>
	/// Base camera class.
	/// </summary>
	class Camera : public RefCounted
	{
	public: 
		Camera() = default;
		Camera(float fov, const glm::vec2& viewportSize, float nearClip, float farClip, bool orthographic = false);
		virtual ~Camera() = default;

		void SetViewportSize(const glm::vec2& viewportSize);
		void SetPosition(const glm::vec3& position);
		void SetPivot(const glm::vec3& pivot);

		glm::mat4& GetViewMatrix();
		glm::mat4& GetProjectionMatrix();
		glm::vec3& GetPosition();
		glm::quat GetOrientation() const;
		glm::vec2 GetViewportSize() const;
		const glm::vec3& GetPivot();

		float GetFOV() const;
		float GetNearClip() const;
		float GetFarClip() const;
	private:
		virtual void UpdateProjection() = 0;
		virtual void UpdateView() = 0;
	protected:
		glm::vec3 GetUpDirection() const;
		glm::vec3 GetRightDirection() const;
		glm::vec3 GetForwardDirection() const;
		glm::vec3 CalculatePosition() const;
	protected:
		glm::vec3 m_Pivot = { 0.0f, 0.0f, 0.0f };
		glm::vec3 m_Position = { 0.0f, 0.0f, 0.0f };
		glm::vec2 m_ViewportSize = { 0.0f, 0.0f };

		glm::mat4 m_ViewMatrix = glm::mat4(1.0f);
		glm::mat4 m_ProjectionMatrix = glm::mat4(1.0f);

		float m_FOV = 0.0f;
		float m_AspectRatio = 0.0f;
		float m_NearClip = 0.0f;
		float m_FarClip = 0.0f;
		float m_Distance = 15.0f;
		float m_Pitch = 0.0f;
		float m_Yaw = 0.0f;

		bool m_Orthographic = false;
	};
}

#endif // !CAMERA_H