#include "pch.h"
#include "ComponentPanel.h"

#include "UI/UI.h"
#include "Utility/FileSystem.h"
#include <imgui_internal.h>

namespace vfd {
	ComponentPanel::ComponentPanel()
	{ }

	static bool dragging = false;
	static glm::vec2 mouseDragStartPos;
	void ComponentPanel::DrawVec3Control(const std::string& label, glm::vec3& values, const std::string& format = "%.2f")
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 0.0f, 0.0f });

		if (ImGui::BeginTable(label.c_str(), 3, ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide)) {
			float width = ImGui::GetContentRegionMax().x / 3 + 1;

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

	void ComponentPanel::DrawIVec3Control(const std::string& label, glm::ivec3& values, const std::string& format)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 0.0f, 0.0f });

		if (ImGui::BeginTable(label.c_str(), 3, ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide)) {
			const float width = ImGui::GetContentRegionMax().x / 3 + 1;

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragInt("##X", &values.x, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragInt("##Y", &values.y, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragInt("##Z", &values.z, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::EndTable();
		}

		ImGui::PopStyleVar();
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

			if (ImGui::Button("Add Component", ImVec2{ ImGui::GetContentRegionAvail().x, 0})) {
				ImGui::OpenPopup("AddComponent");
			}

			if (ImGui::BeginPopup("AddComponent"))
			{
				DrawAddComponentEntry<MeshComponent>("Mesh");
				DrawAddComponentEntry<MaterialComponent>("Material");

				if (m_SelectionContext.HasComponent<DFSPHSimulationComponent>() == false) {
					DrawAddComponentEntry<StaticRigidBodyComponent>("Static Rigidbody");
				}
				ImGui::EndPopup();
			}

			ImGui::Separator();

			DrawComponent<TransformComponent>("Transform Component", [&](auto& component)
			{
				DrawVec3Control("Translation", component.Translation, "%.2f m");
				glm::vec3 rotation = glm::degrees(component.Rotation);
				DrawVec3Control("Rotation", rotation, "%.2f\xc2\xb0");
				component.Rotation = glm::radians(rotation);
				DrawVec3Control("Scale", component.Scale);
			});

			DrawComponent<MeshComponent>("Mesh Component", [&](auto& component)
			{
				Ref<TriangleMesh>& mesh = component.Mesh;

				if (ImGui::Button((mesh ? "Source: " + mesh->GetSourceFilepath() : "Mesh not set").c_str(), ImVec2(ImGui::GetContentRegionMax().x, 0))) {
					const std::string filepath = fs::FileDialog::OpenFile("Mesh files (*.obj)");
					if (filepath.empty() == false) {
						component.Mesh = Ref<TriangleMesh>::Create(filepath);
					}
				}

				if (mesh) {
					UI::ShiftCursor(5, 2);
					ImGui::Text(("Vertices: " + std::to_string(mesh->GetVertexCount())).c_str());
					UI::ShiftCursor(5, 2);
					ImGui::Text(("Triangles: " + std::to_string(mesh->GetTriangleCount())).c_str());
					UI::ShiftCursorY(2);
				}
			});

			DrawComponent<MaterialComponent>("Material Component", [&](auto& component)
			{
				Ref<Material> material = component.Handle;

				ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 0.0f, 0.0f });

				if (material) {
					Ref<Shader> shader = material->GetShader();

					const ShaderLibrary& shaderLib = Renderer::GetShaderLibrary();
					const auto& shaders = shaderLib.GetShaders();

					if (ImGui::BeginCombo("##combo", shader->GetSourceFilepath().c_str())) {
						for (const auto& [key, value] : shaders) {
							const bool isSelected = std::string(key) == shader->GetSourceFilepath().c_str();

							if (ImGui::Selectable(key.c_str(), isSelected)) {
								component.Handle = Ref<Material>::Create(shaderLib.GetShader(key));
							}

							if (isSelected) {
								ImGui::SetItemDefaultFocus();
							}
						}
						ImGui::EndCombo();
					}

					auto& shaderBuffers = shader->GetShaderBuffers();

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
									{
										ImGui::SetNextItemWidth(ImGui::GetContentRegionMax().x - 70);
										int& value = material->GetInt(uniform.GetName());
										ImGui::DragInt("##Z", &value);
										break;
									}
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
									{
										ImGui::SetNextItemWidth(ImGui::GetContentRegionMax().x - 70);
										auto& value = material->GetVector3(uniform.GetName());
										ImGui::ColorEdit3(uniform.GetName().c_str(), glm::value_ptr(value));
										break;
									}
									case ShaderDataType::Float4:
									{
										ImGui::SetNextItemWidth(ImGui::GetContentRegionMax().x - 70);
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
				}
				else {
					const ShaderLibrary& shaderLib = Renderer::GetShaderLibrary();
					const auto& shaders = shaderLib.GetShaders();

					if (ImGui::BeginCombo("##combo", "Shader not set")) {
						for (const auto& [key, value] : shaders) {
							if (ImGui::Selectable(key.c_str(), false)) {
								component.Handle = Ref<Material>::Create(shaderLib.GetShader(key));
							}
						}
						ImGui::EndCombo();
					}
				}

				ImGui::PopStyleVar();
			});

			DrawComponent<SPHSimulationComponent>("SPH Component", [&](auto& component)
			{

			});

			DrawComponent<DFSPHSimulationComponent>("DFSPH Component", [&](auto& component)
			{
			
			});

			DrawComponent<StaticRigidBodyComponent>("Static Rigidbody Component", [&](auto& component)
			{
				Ref<StaticRigidBody> rigidbody = component.RigidBody;
				const StaticRigidBodyDescription& desc = rigidbody->GetDescription();

				//StaticRigidBodyDescription copy = desc;
				//DrawIVec3Control("Resolution", copy.CollisionMapResolution, "%.f");
			});
		}
	}

	void ComponentPanel::OnEvent(Event& event)
	{

	}


}