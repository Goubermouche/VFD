#ifndef EDITOR_CAMERA_H_
#define EDITOR_CAMERA_H_

#include "FluidEngine/Renderer/Camera.h"
#include "FluidEngine/Editor/Panels/ViewportPanel.h"

namespace fe {
	class ViewportPanel;

	class EditorCamera : public Camera
	{
	public:
		EditorCamera(Ref<ViewportPanel> context, float fov, glm::vec2 viewportSize, float nearClip, float farClip);

		void OnEvent(Event& e);
	private:
		virtual void UpdateProjection() override;
		virtual void UpdateView() override;

		bool OnMouseScroll(MouseScrolledEvent& e);
		bool OnMouseMoved(MouseMovedEvent& e);

		void MousePan(const glm::vec2& delta);
		void MouseRotate(const glm::vec2& delta);
		void MouseZoom(float delta);
	private:
		glm::vec2 m_InitialMousePosition = { 0.0f, 0.0f };
		Ref<ViewportPanel> m_Context = nullptr;
	};
}

#endif // !EDITOR_CAMERA_H_