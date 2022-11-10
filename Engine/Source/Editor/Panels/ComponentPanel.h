#ifndef COMPONENT_PANEL_H
#define COMPONENT_PANEL_H

#include "Editor/Panels/EditorPanel.h"

namespace fe {
	class ComponentPanel : public EditorPanel
	{
	public:
		ComponentPanel();
		~ComponentPanel() override = default;

		void OnUpdate() override;
		void OnEvent(Event& event) override;
	private:
		void DrawVec3Control(const std::string& label, glm::vec3& values, const std::string& format);
	private:
	};
}

#endif // !COMPONENT_PANEL_H
