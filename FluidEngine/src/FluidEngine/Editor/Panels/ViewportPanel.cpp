#include "pch.h"
#include "ViewportPanel.h"

#include "FluidEngine/Core/Application.h"
#include "FluidEngine/Compute/GPUCompute.h"


namespace fe {
	ViewportPanel::ViewportPanel()
	{
		Window& win = Application::Get().GetWindow();

		FrameBufferDesc desc;
		desc.width = win.GetWidth();
		desc.height = win.GetHeight();
		desc.attachments = { FrameBufferTextureFormat::RGBA8, FrameBufferTextureFormat::RedInt, FrameBufferTextureFormat::Depth };
		desc.samples = 1;

		m_FrameBuffer = FrameBuffer::Create(desc);
		m_Camera = Ref<EditorCamera>::Create(this, 45.0f, glm::vec2(win.GetWidth(), win.GetHeight()), 0.1f, 1000.0f);
	};


	void ViewportPanel::OnUpdate()
	{
		ImVec2 viewportPanelPosition = ImGui::GetWindowPos();
		ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
		m_Position = ImVec2(viewportPanelPosition.x + contentMin.x, viewportPanelPosition.y + contentMin.y);
		m_Size = ImGui::GetContentRegionAvail();

		uint32_t textureID = m_FrameBuffer->GetColorDescriptionRendererID();
		ImGui::Image((void*)textureID, ImVec2{ m_Size.x, m_Size.y }, ImVec2{ 0.0f, 1.0f }, ImVec2{ 1.0f, 0.0f });

		if (FrameBufferDesc desc = m_FrameBuffer->GetDescription();
			m_Size.x > 0.0f && m_Size.y > 0.0f && // zero sized framebuffer is invalid
			(desc.width != m_Size.x || desc.height != m_Size.y))
		{
			m_FrameBuffer->Resize((uint32_t)m_Size.x, (uint32_t)m_Size.y);
			m_Camera->SetViewportSize({ m_Size.x, m_Size.y });
		}

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
		// Clear frame buffer & prepare it for rendering
		m_FrameBuffer->Bind();
		m_FrameBuffer->ClearAttachment(1, -1);
		Renderer::SetClearColor({ 0.0f, 0.0f, 0.0f, 1.0f });
		Renderer::Clear();

		Renderer::BeginScene(m_Camera);
		m_SceneContext->OnRender();
		OnRender();
		Renderer::EndScene();

		m_FrameBuffer->Unbind();
	}

	void ViewportPanel::OnEvent(Event& e)
	{
		m_Camera->OnEvent(e);
	}

	void ViewportPanel::OnRender()
	{
		
	}
}