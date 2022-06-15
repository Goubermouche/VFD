#include "pch.h"
#include "ViewportPanel.h"

#include "FluidEngine/Core/Application.h"

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

		m_Camera = Ref<EditorCamera>::Create(this, 45, win.GetWidth() / win.GetHeight(), 0.1f, 1000.0f);
		m_Camera->SetViewportSize(win.GetWidth(), win.GetHeight());
	}

	void ViewportPanel::OnUpdate()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });

		if (ImGui::Begin(m_Name.c_str())) {
			// Maybe replace the ImGui::Begin() and ImGui::End() calls with a function inside the editor panel and handle the hover event there? 
			m_Hovered = ImGui::IsWindowHovered();

			m_Size = ImGui::GetContentRegionAvail();

			ImVec2 viewportPanelPosition = ImGui::GetWindowPos();
			ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
			m_Position = ImVec2(viewportPanelPosition.x + contentMin.x, viewportPanelPosition.y + contentMin.y);

			uint32_t textureID = m_FrameBuffer->GetRendererID();
			ImGui::Image((void*)textureID, ImVec2{ m_Size.x, m_Size.y }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
		}
		
		ImGui::End();
		ImGui::PopStyleVar();

		if (FrameBufferDesc desc = m_FrameBuffer->GetSpecification();
			m_Size.x > 0.0f && m_Size.y > 0.0f && // zero sized framebuffer is invalid
			(desc.width != m_Size.x || desc.height != m_Size.y))
		{
			m_FrameBuffer->Resize((uint32_t)m_Size.x, (uint32_t)m_Size.y);
			m_Camera->SetViewportSize(m_Size.x, m_Size.y);
		}

		// clear frame buffer & prepare it for rendering
		m_FrameBuffer->Bind();
		m_FrameBuffer->ClearAttachment(1, -1);

		OnRender();

		m_FrameBuffer->Unbind();
	}

	void ViewportPanel::OnEvent(Event& e)
	{
		m_Camera->OnEvent(e);
	}

	void ViewportPanel::SetSceneContext(Ref<Scene> context)
	{
		m_SceneContext = context;
	}

	void ViewportPanel::OnRender()
	{
		Renderer::SetClearColor({ 1, 0, 0, 1.0f });
		Renderer::Clear();
		Renderer::BeginScene(m_Camera);

		Renderer::SetLineWidth(5);
		Renderer::DrawBox({ 0, 0, 0 }, { 3, 3, 3 }, { 255, 255, 255, 255 });

		Renderer::EndScene();
	}
}