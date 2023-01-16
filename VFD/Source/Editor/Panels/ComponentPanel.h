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

		template<typename T, typename UIFunction>
		void DrawComponentRemovable(const std::string& title, UIFunction function);

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
		static void DrawStringLabel(const std::string& label, const std::string& value, const std::string& format = "%s");
		static void DrawUnsignedIntLabel(const std::string& label, unsigned int value, const std::string& format = "%u");

		template<typename UIFunction>
		static void DrawTreeNode(const char* label, UIFunction function);
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

	template<typename T, typename UIFunction>
	inline void ComponentPanel::DrawComponentRemovable(const std::string& title, UIFunction function)
	{
		if (m_SelectionContext.HasComponent<T>()) {

			bool componentOpen = false;
			if(ImGui::BeginTable(("##" + title).c_str(), 2, ImGuiTableFlags_NoPadInnerX))
			{
				ImGui::TableSetupColumn("##0", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, ImGui::GetContentRegionAvail().x - 60.0f);
				ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, 60.0f);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
			
				componentOpen = ImGui::CollapsingHeader(title.c_str());

				ImGui::TableSetColumnIndex(1);
				if (ImGui::Button("Remove", {60, 21 }))
				{
					m_SelectionContext.RemoveComponent<T>();
					ImGui::EndTable();
					return;
				}

				ImGui::EndTable();
			}

			if (componentOpen) {
				auto& component = m_SelectionContext.GetComponent<T>();
				function(component);
			}
		}
	}

	template<typename UIFunction>
	inline void ComponentPanel::DrawTreeNode(const char* label, UIFunction function)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3.0f, 3.f));
		if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanFullWidth))
		{
			ImGui::PopStyleVar();
			function();
			ImGui::TreePop();
		}
		else
		{
			ImGui::PopStyleVar();
		}
	}
}

#endif // !COMPONENT_PANEL_H
