#include "pch.h"
#include "ComponentPanel.h"

namespace fe {
	ComponentPanel::ComponentPanel()
	{
	}

	void ComponentPanel::OnUpdate()
	{
		if (m_SelectionContext) {
			ImGui::Text(m_SelectionContext.GetComponent<TagComponent>().Tag.c_str());
		}
	}

	void ComponentPanel::OnEvent(Event& event)
	{
	}
}