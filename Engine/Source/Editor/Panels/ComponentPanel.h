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
		template<typename T>
		void DrawAddComponentEntry(const std::string& label) {
			if (!m_SelectionContext.HasComponent<T>())
			{
				if (ImGui::MenuItem(label.c_str()))
				{
					m_SelectionContext.AddComponent<T>();
					ImGui::CloseCurrentPopup();
				}
			}
		}

		template<typename T, typename UIFunction>
		void DrawComponent(const std::string& title, UIFunction function) {
			if (m_SelectionContext.HasComponent<T>()) {
				if (ImGui::CollapsingHeader(title.c_str())) {
					auto& component = m_SelectionContext.GetComponent<T>();
					function(component);
				}
			}
		}

		void DrawVec3Control(const std::string& label, glm::vec3& values, const std::string& format);
	private:
	};
}

#endif // !COMPONENT_PANEL_H
