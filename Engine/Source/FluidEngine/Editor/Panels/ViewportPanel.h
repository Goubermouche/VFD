#ifndef VIEWPORT_PANEL_H
#define VIEWPORT_PANEL_H

#include "FluidEngine/Editor/Panels/EditorPanel.h"
#include "FluidEngine/Renderer/Renderer.h"
#include "FluidEngine/Editor/EditorCamera.h"

namespace fe {
	class EditorCamera;

	class ViewportPanel : public EditorPanel
	{
	public:
		ViewportPanel();

		virtual void OnUpdate() override;
		virtual void OnEvent(Event& e) override;
	private:
		Ref<FrameBuffer> m_FrameBuffer;
		Ref<EditorCamera> m_Camera;

		ImVec2 m_Position;
		ImVec2 m_Size;

		friend class EditorCamera;
	};
}

#endif // !VIEWPORT_PANEL_H