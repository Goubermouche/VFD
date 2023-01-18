#include "pch.h"
#include "ComponentPanel.h"

#include "UI/UI.h"
#include "Utility/FileSystem.h"
#include <imgui_internal.h>

namespace vfd {
	ComponentPanel::ComponentPanel() { }

	static bool dragging = false;
	static glm::vec2 mouseDragStartPos;

	bool ComponentPanel::DrawBoolControl(const std::string& label, bool& value, const std::string& tooltip)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		if (tooltip.empty() == false && ImGui::IsItemHovered())
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 5, 3 });
			ImGui::SetTooltip(tooltip.c_str());
			ImGui::PopStyleVar();
		}
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 22, yPos });

		return ImGui::Checkbox(("##" + label).c_str(), &value);
	}

	bool ComponentPanel::DrawFloatControl(const std::string& label, float& value, float stepSize, const std::string& format, const std::string& tooltip)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		if (tooltip.empty() == false && ImGui::IsItemHovered())
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 5, 3 });
			ImGui::SetTooltip(tooltip.c_str());
			ImGui::PopStyleVar();
		}
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 66, yPos });

		return ImGui::DragFloat(("##" + label).c_str(), &value, stepSize, 0.0f, 0.0f, format.c_str());
	}

	bool ComponentPanel::DrawIntControl(const std::string& label, int& value, const std::string& format, const std::string& tooltip)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		if (tooltip.empty() == false && ImGui::IsItemHovered())
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 5, 3 });
			ImGui::SetTooltip(tooltip.c_str());
			ImGui::PopStyleVar();
		}
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 66, yPos });

		return ImGui::DragInt(("##" + label).c_str(), &value, 1, 0, 0, format.c_str());
	}

	bool ComponentPanel::DrawUnsignedIntControl(const std::string& label, unsigned int& value, const std::string& format, const std::string& tooltip)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		if (tooltip.empty() == false && ImGui::IsItemHovered())
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 5, 3 });
			ImGui::SetTooltip(tooltip.c_str());
			ImGui::PopStyleVar();
		}
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 66, yPos });

		return ImGui::DragScalar(("##" + label).c_str(), ImGuiDataType_U32, &value, 1, 0, 0, format.c_str());
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

	bool ComponentPanel::DrawVec3ControlLabel(const std::string& label, glm::vec3& values, const std::string& format, const std::string& tooltip)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());

		if (tooltip.empty() == false && ImGui::IsItemHovered())
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 5, 3 });
			ImGui::SetTooltip(tooltip.c_str());
			ImGui::PopStyleVar();
		}

		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 200, yPos });

		return ImGui::DragFloat3(("##" + label).c_str(), glm::value_ptr(values), 0.1f, 0.0f, 0.0f, format.c_str());
	}

	bool ComponentPanel::DrawIVec3ControlLabel(const std::string& label, glm::ivec3& values, const std::string& format, const std::string& tooltip)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		if (tooltip.empty() == false && ImGui::IsItemHovered())
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 5, 3 });
			ImGui::SetTooltip(tooltip.c_str());
			ImGui::PopStyleVar();
		}
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 200, yPos });

		return ImGui::DragInt3(("##" + label).c_str(), glm::value_ptr(values), 0.1f, 0.0f, 0.0f, format.c_str());
	}

	bool ComponentPanel::DrawUVec3ControlLabel(const std::string& label, glm::uvec3& values, const std::string& tooltip)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		if (tooltip.empty() == false && ImGui::IsItemHovered())
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 5, 3 });
			ImGui::SetTooltip(tooltip.c_str());
			ImGui::PopStyleVar();
		}
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 200, yPos });

		return ImGui::DragScalarN(("##" + label).c_str(), ImGuiDataType_U32, glm::value_ptr(values), 3);
	}

	void ComponentPanel::DrawFloatLabel(const std::string& label, float value, const std::string& format)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 66, yPos + 3 });
		ImGui::Text(format.c_str(), value);
		UI::ShiftCursorY(5);
	}

	void ComponentPanel::DrawStringLabel(const std::string& label, const std::string& value, const std::string& format)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 66, yPos + 3 });
		ImGui::Text(format.c_str(), value.c_str());
		UI::ShiftCursorY(5);
	}

	void ComponentPanel::DrawUnsignedIntLabel(const std::string& label, unsigned int value, const std::string& format)
	{
		const float yPos = ImGui::GetCursorPos().y;
		UI::ShiftCursor(5, 3);
		ImGui::Text(label.c_str());
		ImGui::SetCursorPos({ ImGui::GetContentRegionMax().x - 66, yPos + 3 });
		ImGui::Text(format.c_str(), value);
		UI::ShiftCursorY(5);
	}

	void ComponentPanel::OnUpdate()
	{
		if (m_SelectionContext) {
			ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 0.0f, 0.0f });
			if (m_SelectionContext.HasComponent<TagComponent>()) {
				auto& tag = m_SelectionContext.GetComponent<TagComponent>().Tag;

				char buffer[ENTITY_NAME_MAX_LENGTH];
				std::strncpy(buffer, tag.c_str(), sizeof(buffer));
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 120.0f);
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

			if (ImGui::Button("Add Component", ImVec2{ 120.0f, 0.0f })) {
				ImGui::OpenPopup("AddComponent");
			}

			if (ImGui::BeginPopup("AddComponent"))
			{
				DrawAddComponentEntry<MeshComponent>("Mesh");
				DrawAddComponentEntry<MaterialComponent>("Material");

				if(EntityHasSimulationComponent(m_SelectionContext) == false)
				{
					DrawAddComponentEntry<RigidBodyComponent>("Rigid Body");
					DrawAddComponentEntry<FluidObjectComponent>("Fluid Object");
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

			DrawComponentRemovable<MeshComponent>("Mesh Component", [&](auto& component)
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

			DrawComponentRemovable<MaterialComponent>("Material Component", [&](auto& component)
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

			DrawComponentRemovable<SPHSimulationComponent>("SPH Component", [&](auto& component)
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

				//bool boundsUpdated = false;
				//boundsUpdated |= DrawVec3ControlLabel("World Min", desc.WorldMin, "%.2f m");
				//boundsUpdated |= DrawVec3ControlLabel("World Max", desc.WorldMax, "%.2f m");

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
					//if (boundsUpdated)
					//{
					//	const glm::vec3 r = desc.WorldMax - desc.WorldMin;
					//	if (r.x > 0.0f && r.y > 0.0f && r.z > 0.0f)
					//	{
					//		component.Handle->UpdateDescription(desc);
					//	}
					//}
					//else
					{
						// component.Handle->UpcdateDescription(desc);
					}
				}
			});

			DrawComponentRemovable<DFSPHSimulationComponent>("DFSPH Component", [&](auto& component)
			{
				DFSPHSimulationDescription desc = component.Handle->GetDescription();
				const bool simulating = component.Handle->GetSimulationState() == DFSPHImplementation::SimulationState::Simulating;

				if(simulating)
				{
					ImGui::BeginDisabled();
				}

				DrawVec3ControlLabel("Gravity", desc.Gravity, "%.2f m/s\xc2\xb2");
				DrawFloatControl("Particle Radius", desc.ParticleRadius, 0.001f, "%.4f");

				DrawTreeNode("Time Step", [&]
				{
					DrawFloatControl("Time Step Size", desc.TimeStepSize, 0.0001f, "%.5f", "Time step size at frame 0");
					DrawFloatControl("Time Step Size Min", desc.MinTimeStepSize, 0.0001f, "%.5f", "Lowest allowed time step size");
				});

				DrawTreeNode("Pressure", [&]
				{
					DrawUnsignedIntControl("Min Pressure Solver Iterations", desc.MinPressureSolverIterations);
					DrawUnsignedIntControl("Max Pressure Solver Iterations", desc.MaxPressureSolverIterations);
					DrawFloatControl("Max Pressure Solver Error", desc.MaxPressureSolverError, 0.5f, "%.1f %%", "Highest allowed pressure solver error");
				});

				DrawTreeNode("Divergence", [&]
				{
					DrawBoolControl("Enable Divergence Solver", desc.EnableDivergenceSolverError);

					if (desc.EnableDivergenceSolverError == false)
					{
						ImGui::BeginDisabled();
					}

					DrawUnsignedIntControl("Min Divergence Solver Iterations", desc.MinDivergenceSolverIterations);
					DrawUnsignedIntControl("Max Divergence Solver Iterations", desc.MaxDivergenceSolverIterations);
					DrawFloatControl("Max Pressure Solver Error", desc.MaxDivergenceSolverError, 0.5f, "%.1f %%", "Highest allowed divergence solver error");

					if (desc.EnableDivergenceSolverError == false)
					{
						ImGui::EndDisabled();
					}
				});

				DrawTreeNode("Viscosity", [&]
				{
					DrawBoolControl("Enable Viscosity Solver", desc.EnableViscositySolver);

					if (desc.EnableViscositySolver == false)
					{
						ImGui::BeginDisabled();
					}

					DrawUnsignedIntControl("Min Viscosity Solver Iterations", desc.MinViscositySolverIterations);
					DrawUnsignedIntControl("Max Viscosity Solver Iterations", desc.MaxViscositySolverIterations);
					DrawFloatControl("Max Viscosity Solver Error", desc.MaxViscositySolverError, 0.5f, "%.1f %%", "Highest allowed viscosity solver error");
					DrawFloatControl("Viscosity", desc.Viscosity, 0.1f, "%.4f", "Particle viscosity coefficient");
					DrawFloatControl("Boundary Viscosity", desc.BoundaryViscosity, 0.1f, "%.4f", "Boundary viscosity coefficient");
					DrawFloatControl("Tangential Distance Factor", desc.TangentialDistanceFactor, 0.01f, "%.2f", "Viscosity friction coefficient");

					if (desc.EnableViscositySolver == false)
					{
						ImGui::EndDisabled();
					}
				});

				DrawTreeNode("Surface Tension", [&]
				{
					DrawBoolControl("Enable Surface Tension Solver", desc.EnableSurfaceTensionSolver);

					if (desc.EnableSurfaceTensionSolver == false)
					{
						ImGui::BeginDisabled();
					}

					DrawUnsignedIntControl("Surface Tension Smooth Passes", desc.SurfaceTensionSmoothPassCount);
					DrawFloatControl("Surface Tension", desc.SurfaceTension, 0.01f, "%.2f", "Surface tension coefficient");
					DrawBoolControl("Temporal Smoothing", desc.TemporalSmoothing, "Interpolate with delta smoothing values");
					DrawIntControl("CSD Fix", desc.CSDFix, "%i", "Sample count per simulation step");
					DrawIntControl("CSD", desc.CSD, "%i", "Sample count per particle per second");

					if (desc.EnableSurfaceTensionSolver == false)
					{
						ImGui::EndDisabled();
					}
				});

				DrawTreeNode("Debug", [&]
				{
					// DrawTreeNode("XXXX", [&] {});
					const DFSPHDebugInfo& info = component.Handle->GetDebugInfo();

					DrawTreeNode("Time Step", [&]
					{
						DrawFloatLabel("Current Time Step", component.Handle->GetCurrentTimeStepSize(), "%.5f ms");
						DrawFloatLabel("Frame Time", info.FrameTime, "%.5f ms");
						DrawUnsignedIntLabel("Frame Index", info.FrameIndex);
						DrawUnsignedIntLabel("Total Frame Count", desc.FrameCount);
					});

					DrawTreeNode("Solver Info", [&]
					{
						DrawFloatLabel("Highest Velocity", component.Handle->GetMaxVelocityMagnitude(), "%.2f m/s\xc2\xb2");
						DrawFloatLabel("Neighborhood Search", info.NeighborhoodSearchTimer.GetElapsed<std::chrono::microseconds>() / 1000.0f, "%.2f ms");
						DrawFloatLabel("Base Solve", info.BaseSolverTimer.GetElapsed<std::chrono::microseconds>() / 1000.0f, "%.2f ms");
						DrawFloatLabel("Surface Tension Solve", info.SurfaceTensionSolverTimer.GetElapsed<std::chrono::microseconds>() / 1000.0f, "%.2f ms");
						ImGui::Separator();
						DrawUnsignedIntLabel("Divergence Solver Iterations", info.DivergenceSolverIterationCount);
						DrawFloatLabel("Divergence Solver Error", info.DivergenceSolverError, "%.2f %");
						DrawFloatLabel("Divergence Solve", info.DivergenceSolverTimer.GetElapsed<std::chrono::microseconds>() / 1000.0f, "%.2f ms");
						ImGui::Separator();
						DrawUnsignedIntLabel("Pressure Solver Iterations", info.PressureSolverIterationCount);
						DrawFloatLabel("Pressure Solver Error", info.PressureSolverError, "%.2f %");
						DrawFloatLabel("Pressure Solve", info.PressureSolverTimer.GetElapsed<std::chrono::microseconds>() / 1000.0f, "%.2f ms");
						ImGui::Separator();
						DrawUnsignedIntLabel("Viscosity Solver Iterations", info.ViscositySolverIterationCount);
						DrawFloatLabel("Viscosity Solver Error", info.ViscositySolverError, "%.2f %");
						DrawFloatLabel("Viscosity Solve", info.ViscositySolverTimer.GetElapsed<std::chrono::microseconds>() / 1000.0f, "%.2f ms");
						ImGui::Separator();
					});

					DrawTreeNode("Rigid Bodies", [&]
					{
						DrawUnsignedIntLabel("Rigid Body Count", component.Handle->GetRigidBodyCount());
						//if(ImGui::Button("Recompute Rigid Bodies", { ImGui::GetContentRegionAvail().x, 20 }))
						//{
						//	
						//}
					});

					DrawTreeNode("Data Size", [&]
					{
						// Particle size, accounts for viscosity solver values 
						constexpr unsigned int particleSize = sizeof(DFSPHParticle) + sizeof(glm::vec3) * 7 + sizeof(glm::mat3x3);
						const unsigned int frameSize = particleSize * component.Handle->GetParticleCount();
						const unsigned int particleSearchSize = component.Handle->GetParticleSearch().GetByteSize();

						DrawStringLabel("Particle Size", fs::FormatFileSize(particleSize));
						DrawStringLabel("Frame Size", fs::FormatFileSize(frameSize));
						DrawStringLabel("Particle Search Size", fs::FormatFileSize(particleSearchSize));
					});
				});
				
				if(desc != component.Handle->GetDescription())
				{
					component.Handle->SetDescription(desc);
				}

				if(ImGui::Button("Simulate", { ImGui::GetContentRegionAvail().x , 30}))
				{
					std::vector<Ref<RigidBody>> rigidBodies;
					std::vector<Ref<FluidObject>> fluidObjects;

					auto& kernel = component.Handle->GetKernel();

					component.Handle->SetDescription(desc);
					auto info = component.Handle->GetInfo();

					for (const entt::entity entity : m_SceneContext->View<FluidObjectComponent, MeshComponent>())
					{
						Entity e = { entity, m_SceneContext.Raw() };

						auto mesh = e.GetComponent<MeshComponent>().Mesh;
						auto fluidObjectDescription = e.GetComponent<FluidObjectComponent>().Description;

						if (mesh == nullptr)
						{
							continue;
						}

						fluidObjectDescription.Mesh = mesh;
						fluidObjectDescription.Transform = m_SceneContext->GetWorldSpaceTransformMatrix(e);

						fluidObjects.push_back(Ref<FluidObject>::Create(fluidObjectDescription, info));
					}

					component.Handle->SetFluidObjects(fluidObjects);
				    info = component.Handle->GetInfo();

					for (const entt::entity entity : m_SceneContext->View<RigidBodyComponent, MeshComponent>()) {
						Entity e = { entity, m_SceneContext.Raw() };

						auto mesh = e.GetComponent<MeshComponent>().Mesh;
						auto rigidBodyDescription = e.GetComponent<RigidBodyComponent>().Description;

						if(mesh == nullptr)
						{
							continue;
						}

						rigidBodyDescription.Mesh = mesh;
						rigidBodyDescription.Transform = m_SceneContext->GetWorldSpaceTransformMatrix(e);

						rigidBodies.push_back(Ref<RigidBody>::Create(rigidBodyDescription, info, kernel));
					}

					component.Handle->SetRigidBodies(rigidBodies);
					component.Handle->Simulate();
				}

				if (simulating)
				{
					ImGui::EndDisabled();
				}
			});

			DrawComponentRemovable<RigidBodyComponent>("Rigidbody Component", [&](auto& component)
			{
				DrawUVec3ControlLabel("Resolution", component.Description.CollisionMapResolution, "Resolution of the precomputed collision map");
				DrawBoolControl("Inverted", component.Description.Inverted);
				DrawFloatControl("Padding", component.Description.Padding, 0.01f, "%.3f", "Collision map padding");
			});

			DrawComponentRemovable<FluidObjectComponent>("Fluid Object Component", [&](auto& component)
			{
				DrawVec3ControlLabel("Velocity", component.Description.Velocity, "%.2f m/s\xc2\xb2");
				DrawUVec3ControlLabel("Resolution", component.Description.Resolution, "Resolution of the generated SDF");
				DrawBoolControl("Inverted", component.Description.Inverted);

				const char* densityValues[] = { "Low Density", "Medium Density", "High Density" };

				if (ImGui::BeginCombo("##combo", densityValues[static_cast<unsigned int>(component.Description.SampleMode)]))
				{
					for (unsigned int i = 0u; i < 3u; i++)
					{
						const bool isSelected = static_cast<unsigned int>(component.Description.SampleMode) == i;

						if (ImGui::Selectable(densityValues[i], isSelected))
						{
							component.Description.SampleMode = static_cast<SampleMode>(i);

							if (isSelected)
							{
								ImGui::SetItemDefaultFocus();
							}
						}
					}

					ImGui::EndCombo();
				}
			});

			ImGui::PopStyleVar();
		}
		else
		{
			const char* message = "Select an entity to view and edit its components";
			ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(message).x) * 0.5f);
			ImGui::Text(message);
		}
	}

	void ComponentPanel::OnEvent(Event& event)
	{

	}

	bool ComponentPanel::EntityHasSimulationComponent(Entity entity)
	{
		return entity.HasComponent<SPHSimulationComponent>();
	}
}