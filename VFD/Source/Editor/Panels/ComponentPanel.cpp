#include "pch.h"
#include "ComponentPanel.h"

#include "UI/UI.h"
#include "Utility/FileSystem.h"
#include <imgui_internal.h>

namespace vfd {
	ComponentPanel::ComponentPanel() { }

	static bool dragging = false;
	static glm::vec2 mouseDragStartPos;

	bool ComponentPanel::DrawFloatControl(const std::string& label, float& value, float stepSize, const std::string& format)
	{
		UI::ShiftCursorX(5);
		ImGui::Text(label.c_str());
		ImGui::SameLine(ImGui::GetWindowWidth() - 66);
		return ImGui::DragFloat(("##" + label).c_str(), &value, stepSize, 0.0f, 0.0f, format.c_str());
	}

	bool ComponentPanel::DrawIntControl(const std::string& label, int& value, const std::string& format)
	{
		UI::ShiftCursorX(5);
		ImGui::Text(label.c_str());
		ImGui::SameLine(ImGui::GetWindowWidth() - 66);
		return ImGui::DragInt(("##" + label).c_str(), &value, 1, 0, 0, format.c_str());
	}

	void ComponentPanel::DrawVec3Control(const std::string& label, glm::vec3& values, const std::string& format)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 0.0f, 0.0f });

		if (ImGui::BeginTable(label.c_str(), 3, ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoReorder | ImGuiTableColumnFlags_NoHide)) {
			const float width = ImGui::GetContentRegionMax().x / 3 + 1;

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragFloat(("##X" + label).c_str(), &values.x, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragFloat(("##Y" + label).c_str(), &values.y, 0.1f, 0.0f, 0.0f, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragFloat(("##Z" + label).c_str(), &values.z, 0.1f, 0.0f, 0.0f, format.c_str());
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
			ImGui::DragInt(("##X" + label).c_str(), &values.x, 1, 0, 0, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragInt(("##Y" + label).c_str(), &values.y, 1, 0, 0, format.c_str());
			ImGui::PopItemWidth();

			ImGui::TableNextColumn();
			ImGui::PushItemWidth(width);
			ImGui::DragInt(("##Z" + label).c_str(), &values.z, 1, 0, 0, format.c_str());
			ImGui::PopItemWidth();

			ImGui::EndTable();
		}

		ImGui::PopStyleVar();
	}

	bool ComponentPanel::DrawVec3ControlLabel(const std::string& label, glm::vec3& values, const std::string& format)
	{
		UI::ShiftCursorX(5);
		ImGui::Text(label.c_str());
		ImGui::SameLine(ImGui::GetWindowWidth() - 200);
		return ImGui::DragFloat3(("##" + label).c_str(), glm::value_ptr(values), 0.1f, 0.0f, 0.0f, format.c_str());
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

				if(EntityHasSimulationComponent(m_SelectionContext) == false)
				{
					DrawAddComponentEntry<StaticRigidBodyComponent>("Static Rigidbody");
					DrawAddComponentEntry<SPHSimulationComponent>("SPH Simulation [Deprecated]");
				}

				ImGui::EndPopup();
			}

			ImGui::Separator();


			// TODO: maybe all components can have their own render function? 
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
				SPHSimulationDescription desc = component.Handle->GetDescription();

				DrawFloatControl("Viscosity", desc.Viscosity);
				DrawFloatControl("Stiffness", desc.Stiffness);
				int stepCount = static_cast<int>(desc.StepCount);
				if(DrawIntControl("Step Count", stepCount))
				{
					desc.StepCount = std::clamp(static_cast<uint32_t>(stepCount), 0u, 20u);
				}
				DrawFloatControl("Time Step", desc.TimeStep, 0.0001f, "%.5f");
				DrawVec3ControlLabel("Gravity", desc.Gravity);
				DrawVec3ControlLabel("World Min", desc.WorldMin, "%.2f m");
				DrawVec3ControlLabel("World Max", desc.WorldMax, "%.2f m");

				if(ImGui::TreeNode("Advanced"))
				{
					DrawFloatControl("Homogeneity", desc.Homogeneity, 0.001f);
					DrawFloatControl("RestDensity", desc.RestDensity);
					DrawFloatControl("GlobalDamping", desc.GlobalDamping);
					DrawFloatControl("BoundsStiffness", desc.BoundsStiffness);
					DrawFloatControl("BoundsDamping", desc.BoundsDamping);
					DrawFloatControl("BoundsDampingCritical", desc.BoundsDampingCritical);
					ImGui::TreePop();
				}

				if(desc != component.Handle->GetDescription())
				{
					component.Handle->UpdateDescription(desc);
				}
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

	bool ComponentPanel::EntityHasSimulationComponent(Entity entity)
	{
		return entity.HasComponent<DFSPHSimulationComponent>() ||
			entity.HasComponent<SPHSimulationComponent>();
	}
}