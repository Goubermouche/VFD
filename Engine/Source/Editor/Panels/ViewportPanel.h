#ifndef VIEWPORT_PANEL_H
#define VIEWPORT_PANEL_H

#include "Editor/Panels/EditorPanel.h"
#include "Renderer/Renderer.h"
#include "Editor/EditorCamera.h"

namespace fe {
	class EditorCamera;

	class ViewportPanel : public EditorPanel
	{
	public:
		ViewportPanel();

		void OnUpdate() override;
		void OnEvent(Event& e) override;
	private:
		Ref<FrameBuffer> m_FrameBuffer;
		Ref<EditorCamera> m_Camera;

		ImVec2 m_Position;
		ImVec2 m_Size;

		friend class EditorCamera;
	};
}

#endif // !VIEWPORT_PANEL_H