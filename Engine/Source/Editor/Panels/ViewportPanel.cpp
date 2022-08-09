#include "pch.h"
#include "ViewportPanel.h"

#include "Core/Application.h"
#include "Core/Time.h"
#include  "UI/UI.h"

namespace fe {
	ViewportPanel::ViewportPanel()
	{
		const Window& win = Application::Get().GetWindow();

		m_Camera = Ref<EditorCamera>::Create(this, 45.0f, glm::vec2(win.GetWidth(), win.GetHeight()), 0.1f, 1000.0f);
		m_Camera->SetPosition({ 10, 10, 10 }); // Set default camera position

		FrameBufferDesc desc;
		desc.Width = win.GetWidth();
		desc.Height = win.GetHeight();
		desc.Samples = 1;

		desc.Attachments = {
			TextureFormat::RGBA8,
			TextureFormat::RedInt,
			TextureFormat::Depth
		};

		m_FrameBuffer = Ref<FrameBuffer>::Create(desc);
	};

	void ViewportPanel::OnUpdate()
	{
		const ImVec2 viewportPanelPosition = ImGui::GetWindowPos();
		const ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
		const ImVec2& pos = ImVec2(viewportPanelPosition.x + contentMin.x, viewportPanelPosition.y + contentMin.y);
		const ImVec2 size = ImGui::GetContentRegionAvail();
		m_Position = { pos.x, pos.y };
		m_Size = { size.x, size.y };

		UI::Image(m_FrameBuffer->GetAttachment(0), { m_Size.x, m_Size.y }, {0.0f, 1.0f}, {1.0f, 0.0f});

		if (const FrameBufferDesc& desc = m_FrameBuffer->GetDescription();
			desc.Width > 0.0f && desc.Height > 0.0f &&
			(desc.Width != m_Size.x || desc.Height != m_Size.y))
		{
			m_FrameBuffer->Resize(m_Size.x, m_Size.y);
			m_Camera->SetViewportSize({ m_Size.x, m_Size.y });
		}

		// TEMP: context menu test
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 8.0f, 6.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 2.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4, 4.0f });

			ImGui::PushStyleColor(ImGuiCol_PopupBg, (ImU32)UI::Description.ContextMenuBackground);
			ImGui::PushStyleColor(ImGuiCol_Header, (ImU32)UI::Description.Transparent);
			ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImU32)UI::Description.Transparent);
			ImGui::PushStyleColor(ImGuiCol_HeaderActive, (ImU32)UI::Description.Transparent);
			ImGui::PushStyleColor(ImGuiCol_Border, (ImU32)UI::Description.ContextMenuBorder);
			ImGui::PushStyleColor(ImGuiCol_Separator, (ImU32)UI::Description.ContextMenuBorder);

			if (ImGui::BeginPopupContextWindow(nullptr, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
				if (UI::MenuItem("Shade Smooth")) {

				}
				UI::ShiftCursorY(2);
				if (UI::MenuItem("Shade Flat")) {

				}
				UI::Separator();
				// ------------------------------
				if (UI::BeginMenu("Convert To")) {

					if (UI::MenuItem("Curve")) {

					}
					UI::ShiftCursorY(2);
					if (UI::MenuItem("Mesh")) {

					}
					UI::ShiftCursorY(2);
					if (UI::MenuItem("Grease Pencil")) {

					}
					ImGui::EndMenu();
				}
				UI::ShiftCursorY(2);
				if (UI::BeginMenu("Set Origin")) {
					if (UI::MenuItem("Geometry to Origin")) {

					}
					UI::ShiftCursorY(2);
					if (UI::MenuItem("Origin to Geometry")) {

					}
					UI::ShiftCursorY(2);
					if (UI::MenuItem("Origin to 3D Cursor")) {

					}
					ImGui::EndMenu();
				}
				// ------------------------------
				UI::Separator();
				if (UI::MenuItem("Copy Objects", "Ctrl C")) {

				}
				UI::ShiftCursorY(2);
				if (UI::MenuItem("Paste Objects", "Ctrl V")) {

				}
				// ------------------------------
				UI::Separator();
				if (UI::MenuItem("Duplicate Objects", "Shift D")) {

				}
				UI::ShiftCursorY(2);
				if (UI::MenuItem("Duplicate Linked", "Alt D")) {

				}
				ImGui::EndPopup();
			}

			ImGui::PopStyleVar(3);
			ImGui::PopStyleColor(6);
		}

		m_FrameBuffer->Bind();

		Renderer::SetClearColor({ 0, 0, 0, 1.0f });
		Renderer::Clear();
		Renderer::BeginScene(m_Camera);

		m_SceneContext->OnRender();

		// Renderer::SetLineWidth(2);
		// Renderer::DrawPoint({ -2.0f, 0.0f, 2.0f }, { 1.0f, 0.0f, 1.0f, 1.0f }, std::sinf(Time::Get()) * 10.0f + 10.0f);
		// Renderer::DrawLine({ -2.0f, 0.0f, 2.0f }, { 2.0f, 0.0f, 2.0f }, { 0.0f, 1.0f, 1.0f, 1.0f });
		// Renderer::DrawQuad(glm::mat4(1.0f), { 1.0f, 0.0f, 0.0f, 1.0f });
		// Renderer::DrawQuad(glm::translate(glm::mat4(1.0f), { 1, 0, 0 }), { 0.0f, 1.0f, 0.0f, 1.0f });
		// Renderer::DrawQuad(glm::translate(glm::mat4(1.0f), {1, 1, 0}), {0.0f, 0.0f, 1.0f, 1.0f});
		// Renderer::DrawBox(glm::translate(glm::mat4(1.0f), {0.5f, 0.5f, 0.0f}), {1.0f, 1.0f, 1.0f, 1.0f});

		// Renderer::DrawLine({ -9999, 0, 0 }, { 9999, 0, 0 }, { 0.965,0.212,0.322 , 1 });
		// Renderer::DrawLine({ 0, 0, -9999 }, { 0, 0, 9999 }, { 0.498,0.773,0.067 , 1});

		Renderer::EndScene();
		if (ImGui::IsMouseClicked(0) && ImGui::IsWindowHovered()) {
			const glm::vec2 panelSpace{ Input::GetMouseX() - m_Position.x , Input::GetMouseY() - m_Position.y };
			const glm::vec2 textureSpace = { panelSpace.x, m_Size.y - panelSpace.y };

			const uint32_t pixelData = m_FrameBuffer->ReadPixel(1, textureSpace.x, textureSpace.y);
			// ERR(pixelData);

			const Entity entity = m_SceneContext->TryGetEntityWithUUID(pixelData);
			Editor::Get().SetSelectionContext(entity);
		}

		m_FrameBuffer->Unbind();
	}

	void ViewportPanel::OnEvent(Event& e)
	{
		m_Camera->OnEvent(e);
	}
}
