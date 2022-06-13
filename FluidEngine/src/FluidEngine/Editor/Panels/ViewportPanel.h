#ifndef VIEWPORT_PANEL_H
#define VIEWPORT_PANEL_H

#include "FluidEngine/Editor/Panels/EditorPanel.h"

namespace fe {
	class ViewportPanel : public EditorPanel
	{
	public:
		ViewportPanel();

		virtual void OnUpdate() override;
		virtual void OnEvent(Event& e) override;
		virtual void SetSceneContext(Ref<Scene> context) override;
	private:

	};
}

#endif // !VIEWPORT_PANEL_H