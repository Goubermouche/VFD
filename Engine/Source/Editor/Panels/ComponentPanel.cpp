#include "pch.h"
#include "ComponentPanel.h"

#include "UI/UI.h"
#include "Utility/FileSystem.h"
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

		if (ImGui::BeginTable(label.c_str(), 3, ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide)) {
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
				Ref<Material> material = component.Handle;
				Ref<Shader> shader = material->GetShader();
				auto& shaderBuffers = shader->GetShaderBuffers();

				ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 0.0f, 0.0f });

				ImGui::Text(("Source: " + FilenameFromFilepath(shader->GetSourceFilepath())).c_str());
				UI::ShiftCursorY(4);

				for (auto& buffer : shaderBuffers)
				{
					if (buffer.IsPropertyBuffer) {
						for (auto& [key, uniform] : buffer.Uniforms)
						{
							if (ImGui::BeginTable("##material", 2)) {
								ImGui::TableSetupColumn("##0", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide, 70);
								ImGui::TableNextColumn();
								UI::ShiftCursor(5, 2);
								ImGui::Text(uniform.GetName().c_str());
								UI::ShiftCursorY(-2);

								ImGui::TableNextColumn();
								switch (uniform.GetType())
								{
								case ShaderDataType::Bool:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::Int:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::Uint:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::Float:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::Float2:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::Float3:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::Float4:
								{
									ImGui::SetNextItemWidth(ImGui::GetWindowWidth() - 70);
									auto& value = material->GetVector4(uniform.GetName());
									ImGui::ColorEdit4(uniform.GetName().c_str(), glm::value_ptr(value));
									break;
								}
								case ShaderDataType::Mat3:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::Mat4:
									ASSERT("Not implemented!");
									break;
								case ShaderDataType::None:
									ASSERT("Not implemented!");
									break;
								}

								ImGui::EndTable();
							}
						}
					}
				}

				ImGui::PopStyleVar();
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