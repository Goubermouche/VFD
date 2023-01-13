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
		static bool EntityHasSimulationComponent(Entity entity);
		template<typename T>
		void DrawAddComponentEntry(const std::string& label);

		template<typename T, typename UIFunction>
		void DrawComponent(const std::string& title, UIFunction function);

		static bool DrawBoolControl(const std::string& label, bool& value, const std::string& tooltip = "");
		static bool DrawFloatControl(const std::string& label, float& value, float stepSize = 0.1f, const std::string& format = "%.2f", const std::string& tooltip = "");
		static bool DrawIntControl(const std::string& label, int& value, const std::string& format = "%i", const std::string& tooltip = "");
		static bool DrawUnsignedIntControl(const std::string& label, unsigned int& value, const std::string& format = "%i", const std::string& tooltip = "");
		static void DrawVec3Control(const std::string& label, glm::vec3& values, const std::string& format = "%.2f");
		static void DrawIVec3Control(const std::string& label, glm::ivec3& values, const std::string& format = "%i");
		static bool DrawVec3ControlLabel(const std::string& label, glm::vec3& values, const std::string& format = "%.2f", const std::string& tooltip = "");
		static bool DrawIVec3ControlLabel(const std::string& label, glm::ivec3& values, const std::string& format = "%i", const std::string& tooltip = "");
		static bool DrawUVec3ControlLabel(const std::string& label, glm::uvec3& values, const std::string& tooltip = "");

		static void DrawFloatLabel(const std::string& label, float value, const std::string& format = "%.2f");
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
