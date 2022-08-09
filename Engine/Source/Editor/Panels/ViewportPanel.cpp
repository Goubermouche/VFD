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
		const auto& pos = ImVec2(viewportPanelPosition.x + contentMin.x, viewportPanelPosition.y + contentMin.y);
		const auto size = ImGui::GetContentRegionAvail();
		m_Position = {pos.x, pos.y};
		m_Size = { size.x, size.y };

		const uint32_t textureID = m_FrameBuffer->GetAttachmentRendererID(0);
		ImGui::Image((void*)textureID, ImVec2{ m_Size.x, m_Size.y }, ImVec2{ 0.0f, 1.0f }, ImVec2{ 1.0f, 0.0f });
		
		if (const auto& desc = m_FrameBuffer->GetDescription();
			desc.Width > 0.0f && desc.Height > 0.0f &&
			(desc.Width != m_Size.x || desc.Height != m_Size.y))
		{
			m_FrameBuffer->Resize(m_Size.x, m_Size.y);
			m_Camera->SetViewportSize({ m_Size.x, m_Size.y });
		}

		// TEMP 
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 2.0f, 2.0f });
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4.0f, 4.0f });
			if (ImGui::BeginPopupContextWindow(nullptr, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
				if (ImGui::MenuItem("Viewport context menu test")) {
				}
				ImGui::Separator();
				if (ImGui::MenuItem("Shade smooth")) {
				}
				if (ImGui::MenuItem("Shade flat")) {
				}
				ImGui::EndPopup();
			}

			ImGui::PopStyleVar(2);
		}

		m_FrameBuffer->Bind();

		Renderer::SetClearColor({ 0, 0, 0, 1.0f });
		Renderer::Clear();
		Renderer::BeginScene(m_Camera);

		m_SceneContext->OnRender();

		// Renderer::SetLineWidth(2);
		//Renderer::DrawPoint({ -2.0f, 0.0f, 2.0f }, { 1.0f, 0.0f, 1.0f, 1.0f }, std::sinf(Time::Get()) * 10.0f + 10.0f);
		//Renderer::DrawLine({ -2.0f, 0.0f, 2.0f }, { 2.0f, 0.0f, 2.0f }, { 0.0f, 1.0f, 1.0f, 1.0f });
		//Renderer::DrawQuad(glm::mat4(1.0f), { 1.0f, 0.0f, 0.0f, 1.0f });
		//Renderer::DrawQuad(glm::translate(glm::mat4(1.0f), { 1, 0, 0 }), { 0.0f, 1.0f, 0.0f, 1.0f });
		//Renderer::DrawQuad(glm::translate(glm::mat4(1.0f), {1, 1, 0}), {0.0f, 0.0f, 1.0f, 1.0f});
		//Renderer::DrawBox(glm::translate(glm::mat4(1.0f), {0.5f, 0.5f, 0.0f}), {1.0f, 1.0f, 1.0f, 1.0f});

		//Renderer::DrawLine({ -9999, 0, 0 }, { 9999, 0, 0 }, { 0.965,0.212,0.322 , 1 });
		//Renderer::DrawLine({ 0, 0, -9999 }, { 0, 0, 9999 }, { 0.498,0.773,0.067 , 1});

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
