#ifndef COMPONENT_PANEL_H
#define COMPONENT_PANEL_H

#include "Editor/Panels/EditorPanel.h"

namespace vfd {
	class ComponentPanel : public EditorPanel
	{
	public:
		ComponentPanel();
		~ComponentPanel() override = default;

		void OnUpdate() override;
		void OnEvent(Event& event) override;
	private:
		template<typename T>
		void DrawAddComponentEntry(const std::string& label);

		template<typename T, typename UIFunction>
		void DrawComponent(const std::string& title, UIFunction function);

		void DrawVec3Control(const std::string& label, glm::vec3& values, const std::string& format);
		void DrawIVec3Control(const std::string& label, glm::ivec3& values, const std::string& format);
	};

	template<typename T>
	inline void ComponentPanel::DrawAddComponentEntry(const std::string& label)
	{
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
	inline void ComponentPanel::DrawComponent(const std::string& title, UIFunction function)
	{
		if (m_SelectionContext.HasComponent<T>()) {
			if (ImGui::CollapsingHeader(title.c_str())) {
				auto& component = m_SelectionContext.GetComponent<T>();
				function(component);
			}
		}
	}
}

#endif // !COMPONENT_PANEL_H