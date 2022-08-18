#include "pch.h"
#include "ViewportPanel.h"

#include "Core/Application.h"
#include "Core/Time.h"
#include  "UI/UI.h"

namespace fe {
	ViewportPanel::ViewportPanel()
	{
		const Window& win = Application::Get().GetWindow();

		// Camera
		m_Camera = Ref<EditorCamera>::Create(this, 50.0f, glm::vec2(win.GetWidth(), win.GetHeight()), 0.1f, 1000.0f, false);
		m_Camera->SetPosition({ 10, 5,5 }); // Set default camera position

		// Frame buffer
		FrameBufferDesc desc;
		desc.Width = win.GetWidth();
		desc.Height = win.GetHeight();
		desc.Samples = 4;

		desc.Attachments = {
			TextureFormat::RGBA8,
			/*TextureFormat::RedInt,*/
			TextureFormat::Depth
		};

		m_FrameBuffer = Ref<FrameBuffer>::Create(desc);

		// Grid
		std::vector<float> vertices = {
			-1.0f, -1.0f, 0.0f, 
			 1.0f, -1.0f, 0.0f, 
			 1.0f,  1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f
		};

		std::vector<uint32_t> indices = {
			0, 1, 2,
			2, 3, 0
		};

		m_GridVAO = Ref<VertexArray>::Create();

		const Ref<IndexBuffer> gridIndexBuffer = Ref<IndexBuffer>::Create(indices);
		Ref<VertexBuffer> qridVertexBuffer = Ref<VertexBuffer>::Create(vertices);
		qridVertexBuffer->SetLayout({ {ShaderDataType::Float3, "a_Position"} });

		m_GridVAO->AddVertexBuffer(qridVertexBuffer);
		m_GridVAO->SetIndexBuffer(gridIndexBuffer);

		m_GridMaterial = Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/GridPlaneShader.glsl"));
		m_GridMaterial->Set("color", { 0.1f, 0.1f , 0.1f });
		m_GridMaterial->Set("scale", 0.1f);
		m_GridMaterial->Set("near", m_Camera->GetNearClip());
		m_GridMaterial->Set("far", m_Camera->GetFarClip());
	};

	void ViewportPanel::OnUpdate()
	{
		const ImVec2 viewportPanelPosition = ImGui::GetWindowPos();
		const ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
		const ImVec2& pos = ImVec2(viewportPanelPosition.x + contentMin.x, viewportPanelPosition.y + contentMin.y);
		const ImVec2 size = ImGui::GetContentRegionAvail();
		m_Position = { pos.x, pos.y };
		m_Size = { size.x, size.y };

		UI::Image(m_FrameBuffer->GetAttachment(0), { m_Size.x, m_Size.y }, { 0.0f, 1.0f }, { 1.0f, 0.0f });

		// Resize the frame buffer when the panel size changes 
		if (const FrameBufferDesc& desc = m_FrameBuffer->GetDescription();
			desc.Width > 0.0f && desc.Height > 0.0f &&
			(desc.Width != m_Size.x || desc.Height != m_Size.y))
		{
			m_FrameBuffer->Resize(m_Size.x, m_Size.y);
			m_Camera->SetViewportSize({ m_Size.x, m_Size.y });
		}

		// Context menu
		{
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 4, 4.0f });

			if (ImGui::BeginPopupContextWindow(nullptr, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
				if (ImGui::MenuItem("Open Scene")) {
					Editor::Get().LoadSceneContext();
				}

				if (ImGui::MenuItem("Save Scene", "Ctrl S", false, false)) {
					Editor::Get().SaveCurrentSceneContext();
				}
				ImGui::EndPopup();
			}

			ImGui::PopStyleVar();
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

		// Grid
		Renderer::DrawTrianglesIndexed(m_GridVAO, m_GridVAO->GetIndexBuffer()->GetCount(), m_GridMaterial);

		m_FrameBuffer->Unbind();
	}

	void ViewportPanel::OnEvent(Event& event)
	{
		m_Camera->OnEvent(event);
	}
}