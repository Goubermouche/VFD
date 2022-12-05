#include "pch.h"
#include "ViewportPanel.h"

#include "Core/Application.h"
#include "UI/UI.h"

namespace vfd {
	ViewportPanel::ViewportPanel()
	{
		const Window& win = Application::Get().GetWindow();

		// Camera
		m_Camera = Ref<EditorCamera>::Create(this, 50.0f, glm::vec2(win.GetWidth(), win.GetHeight()), 0.1f, 700.0f, 1.8f, CameraType::Perspective);
		m_Camera->SetPosition({ 4, 4, 4 }); // Set default camera position
		m_Camera->SetPivot({ 0.0f, 0.0, 0.0f });
		// Frame buffer
		FrameBufferDescription desc;
		desc.Width = win.GetWidth();
		desc.Height = win.GetHeight();
		desc.Samples = 1;

		desc.Attachments = {
			TextureFormat::RGBA8,
			TextureFormat::RedInt,
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
			0u, 1u, 2u,
			2u, 3u, 0u
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
		m_GridMaterial->Set("far", m_Camera->GetFarClip() * 8.f);
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
		if (const FrameBufferDescription& desc = m_FrameBuffer->GetDescription();
			desc.Width > 0 && desc.Height > 0 &&
			(desc.Width != static_cast<uint32_t>(m_Size.x) || desc.Height != static_cast<uint32_t>(m_Size.y)))
		{
			m_FrameBuffer->Resize(static_cast<uint32_t>(m_Size.x), static_cast<uint32_t>(m_Size.y));
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

		Renderer::EndScene();

		// Sample FBO
		// NOTE: this has to be called before drawing the grid, since the grid shader will override 
		//       every component of the fragments "below" it. 
		//if (ImGui::IsMouseClicked(0) && m_Hovered)
		//{
		//	glm::vec2 panelSpace{ Input::GetMouseX() - m_Position.x , Input::GetMouseY() - m_Position.y };
		//	glm::vec2 textureSpace = { panelSpace.x, m_Size.y - panelSpace.y };

		//	uint32_t pixelData = m_FrameBuffer->ReadPixel(1, textureSpace.x, textureSpace.y);

		//	Entity entity = m_SceneContext->TryGetEntityWithUUID(pixelData);
		//	Editor::Get().SetSelectionContext(entity);
		//}

		// Render grid
		Renderer::DrawTrianglesIndexed(m_GridVAO, m_GridVAO->GetIndexBuffer()->GetCount(), m_GridMaterial);

		m_FrameBuffer->Unbind();
	}

	void ViewportPanel::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<SceneSavedEvent>(BIND_EVENT_FN(OnSceneSaved));
		dispatcher.Dispatch<SceneLoadedEvent>(BIND_EVENT_FN(OnSceneLoaded));

		m_Camera->OnEvent(event);
	} 

	bool ViewportPanel::OnSceneSaved(SceneSavedEvent& event)
	{
		// TODO: add support for multiple panels
		SceneData& data = m_SceneContext->GetData();

		data.CameraPosition = m_Camera->GetPosition();
		data.CameraPivot = m_Camera->GetPivot();

		return false;
	}

	bool ViewportPanel::OnSceneLoaded(SceneLoadedEvent& event)
	{
		// TODO: add support for multiple panels
		const SceneData& data = m_SceneContext->GetData();

		m_Camera->SetPosition(data.CameraPosition);
		m_Camera->SetPivot(data.CameraPivot);

		return false;
	}
}