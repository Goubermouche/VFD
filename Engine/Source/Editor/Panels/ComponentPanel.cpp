#include "pch.h"
#include "ComponentPanel.h"
#include <imgui_internal.h>

namespace fe {
	ComponentPanel::ComponentPanel()
	{
	}

	static bool dragging = false;
	static glm::vec2 mouseDragStartPos;
	void ComponentPanel::DrawVec3Control(const std::string& label, glm::vec3& values, const std::string& format = "%.2f")
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 0.0f, 0.0f });

		if (ImGui::BeginTable(label.c_str(), 3)) {
			float width = ImGui::GetWindowWidth() / 3;

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::EndTable();
		}

		ImGui::PopStyleVar();
	}

	template<typename T, typename UIFunction>
	void DrawComponent(const std::string& title, Entity entity, UIFunction function) {
		if (entity.HasComponent<T>()) {
			if (ImGui::CollapsingHeader(title.c_str())) {
				auto& component = entity.GetComponent<T>();
				function(component);
			}
		}
	}

	void ComponentPanel::OnUpdate()
	{
		if (m_SelectionContext) {
			if (m_SelectionContext.HasComponent<TagComponent>()) {
				auto& tag = m_SelectionContext.GetComponent<TagComponent>().Tag;

				char buffer[ENTITY_NAME_MAX_LENGTH];
				std::strncpy(buffer, tag.c_str(), sizeof(buffer));
				ImGui::InputText("##Tag", buffer, sizeof(buffer));

				if (ImGui::IsItemDeactivatedAfterEdit())
				{
					if (strlen(buffer) != 0) {
						tag = std::string(buffer);
					}
				}
			}

			ImGui::SameLine();
			ImGui::PushItemWidth(-1);

			if (ImGui::Button("Add Component")) {
			}

			ImGui::Separator();

			DrawComponent<TransformComponent>("Transform Component", m_SelectionContext, [&](auto& component)
			{
				DrawVec3Control("Translation", component.Translation, "%.2f m");
				glm::vec3 rotation = glm::degrees(component.Rotation);
				DrawVec3Control("Rotation", rotation, "%.2f\xc2\xb0");
				component.Rotation = glm::radians(rotation);
				DrawVec3Control("Scale", component.Scale);
			});

			DrawComponent<MeshComponent>("Mesh Component", m_SelectionContext, [&](auto& component)
			{

			});

			DrawComponent<MaterialComponent>("Material Component", m_SelectionContext, [&](auto& component)
			{

			});

			DrawComponent<SPHSimulationComponent>("SPH Component", m_SelectionContext, [&](auto& component)
			{

			});

			DrawComponent<DFSPHSimulationComponent>("DFSPH Component", m_SelectionContext, [&](auto& component)
			{

			});
		}
	}

	void ComponentPanel::OnEvent(Event& event)
	{

	}


}