#include "pch.h"
#include "Renderer.h"

#include "FluidEngine/Platform/OpenGL/OpenGLRenderer.h"

namespace fe {
	RendererAPI* Renderer::s_RendererAPI = nullptr;
	RendererData Renderer::s_Data = RendererData();
	Ref<EditorCamera> Renderer::s_Camera = nullptr;

	void Renderer::Init()
	{
		s_RendererAPI = new opengl::OpenGLRenderer;
		s_RendererAPI->Init();

		// Initialize batch renderer
		s_Data.lineVertexArray = VertexArray::Create();

		s_Data.lineVertexBuffer = VertexBuffer::Create(s_Data.maxVertices * sizeof(LineVertex));
		s_Data.lineVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    }
		});

		s_Data.lineVertexArray->AddVertexBuffer(s_Data.lineVertexBuffer);
		s_Data.lineVertexBufferBase = new LineVertex[s_Data.maxVertices];

		s_Data.lineMaterial = Material::Create(Shader::Create("res/Shaders/LineShader.glsl"));

		LOG("renderer initialized successfully");
	}

	void Renderer::BeginScene(Ref<EditorCamera> camera)
	{
		s_Camera = camera;

		s_Data.lineMaterial->Set("view", s_Camera->GetViewMatrix());
		s_Data.lineMaterial->Set("proj", s_Camera->GetProjectionMatrix());

		StartBatch();
	}

	void Renderer::EndScene()
	{
		Flush();
	}

	void Renderer::SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
	{
		s_RendererAPI->SetViewport(x, y, width, height);
	}

	void Renderer::SetClearColor(const glm::vec4& color)
	{
		s_RendererAPI->SetClearColor(color);
	}

	void Renderer::Clear()
	{
		s_RendererAPI->Clear();
	}

	void Renderer::DrawLine(const glm::vec3& p0, const glm::vec3& p1, const glm::vec4& color)
	{
		s_Data.lineVertexBufferPtr->position = p0;
		s_Data.lineVertexBufferPtr->color = color;
		s_Data.lineVertexBufferPtr++;

		s_Data.lineVertexBufferPtr->position = p1;
		s_Data.lineVertexBufferPtr->color = color;
		s_Data.lineVertexBufferPtr++;

		s_Data.lineVertexCount += 2;
	}

	void Renderer::DrawBox(const glm::vec3& position, const glm::vec3& size, const glm::vec4& color)
	{
		float halfX = size.x / 2;
		float halfY = size.y / 2;
		float halfZ = size.z / 2;

		DrawLine({ position.x - halfX, position.y - halfY, position.z - halfZ }, { position.x - halfX, position.y - halfY, position.z + halfZ }, color);
		DrawLine({ position.x + halfX, position.y - halfY, position.z - halfZ }, { position.x + halfX, position.y - halfY, position.z + halfZ }, color);
		DrawLine({ position.x - halfX, position.y - halfY, position.z - halfZ }, { position.x + halfX, position.y - halfY, position.z - halfZ }, color);
		DrawLine({ position.x - halfX, position.y - halfY, position.z + halfZ }, { position.x + halfX, position.y - halfY, position.z + halfZ }, color);

		DrawLine({ position.x - halfX, position.y + halfY, position.z - halfZ }, { position.x - halfX, position.y + halfY, position.z + halfZ }, color);
		DrawLine({ position.x + halfX, position.y + halfY, position.z - halfZ }, { position.x + halfX, position.y + halfY, position.z + halfZ }, color);
		DrawLine({ position.x - halfX, position.y + halfY, position.z - halfZ }, { position.x + halfX, position.y + halfY, position.z - halfZ }, color);
		DrawLine({ position.x - halfX, position.y + halfY, position.z + halfZ }, { position.x + halfX, position.y + halfY, position.z + halfZ }, color);

		DrawLine({ position.x - halfX, position.y - halfY, position.z - halfZ }, { position.x - halfX, position.y + halfY, position.z - halfZ }, color);
		DrawLine({ position.x - halfX, position.y - halfY, position.z + halfZ }, { position.x - halfX, position.y + halfY, position.z + halfZ }, color);
		DrawLine({ position.x + halfX, position.y - halfY, position.z - halfZ }, { position.x + halfX, position.y + halfY, position.z - halfZ }, color);
		DrawLine({ position.x + halfX, position.y - halfY, position.z + halfZ }, { position.x + halfX, position.y + halfY, position.z + halfZ }, color);
	}

	float Renderer::GetLineWidth()
	{
		return s_Data.lineWidth;
	}

	void Renderer::SetLineWidth(float width)
	{
		s_Data.lineWidth = width;
	}

	void Renderer::StartBatch()
	{
		s_Data.lineVertexCount = 0;
		s_Data.lineVertexBufferPtr = s_Data.lineVertexBufferBase;
	}

	void Renderer::NextBatch()
	{
		Flush();
		StartBatch();
	}

	void Renderer::Flush()
	{
		if (s_Data.lineVertexCount)
		{
			uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.lineVertexBufferPtr - (uint8_t*)s_Data.lineVertexBufferBase);
			s_Data.lineVertexBuffer->SetData(s_Data.lineVertexBufferBase, dataSize);

			s_Data.lineMaterial->Bind();
			s_RendererAPI->SetLineWidth(s_Data.lineWidth);
			s_RendererAPI->DrawLines(s_Data.lineVertexArray, s_Data.lineVertexCount);
		}
	}
}