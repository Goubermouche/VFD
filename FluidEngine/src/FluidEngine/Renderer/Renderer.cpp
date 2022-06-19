#include "pch.h"
#include "Renderer.h"

#include "FluidEngine/Platform/OpenGL/OpenGLRenderer.h"

namespace fe {
	RendererAPI* Renderer::s_RendererAPI = nullptr;
	RendererData Renderer::s_Data = RendererData();
	Ref<Camera> Renderer::s_Camera = nullptr;

	void Renderer::Init()
	{
		s_RendererAPI = new opengl::OpenGLRenderer;
		s_RendererAPI->Init();

		// Initialize the batch renderer
		// Points
		s_Data.pointVertexArray = VertexArray::Create();
		s_Data.pointVertexBuffer = VertexBuffer::Create(s_Data.maxVertices * sizeof(PointVertex));
		s_Data.pointVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    },
			{ ShaderDataType::Float,  "a_Radius"   }
		});
		s_Data.pointVertexArray->AddVertexBuffer(s_Data.pointVertexBuffer);
		s_Data.pointVertexBufferBase = new PointVertex[s_Data.maxVertices];
	    s_Data.pointMaterial = Material::Create(Shader::Create("res/Shaders/PointShaderDiffuse.glsl"));
		// s_Data.pointMaterial = Material::Create(Shader::Create("res/Shaders/PointShader.glsl"));

		// Lines
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

	void Renderer::BeginScene(Ref<Camera> camera)
	{
		s_Camera = camera;

		// Points
		s_Data.pointMaterial->Set("view", s_Camera->GetViewMatrix());
		s_Data.pointMaterial->Set("proj", s_Camera->GetProjectionMatrix());
		s_Data.pointMaterial->Set("viewportSize", s_Camera->GetViewportSize());

		// Lines
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

	void Renderer::DrawPoint(const glm::vec3& p, const glm::vec4 color, float radius)
	{
		s_Data.pointVertexBufferPtr->position = p;
		s_Data.pointVertexBufferPtr->color = color;
		s_Data.pointVertexBufferPtr->radius = radius;
		s_Data.pointVertexBufferPtr++;

		s_Data.pointVertexCount++;
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
		// the performance here could be improved by adding a batched cube renderer, however,
		// at this point in time this works just fine.

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
		// Points
		s_Data.pointVertexCount = 0;
		s_Data.pointVertexBufferPtr = s_Data.pointVertexBufferBase;

		// Lines
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
		// Points
		if (s_Data.pointVertexCount)
		{
			uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.pointVertexBufferPtr - (uint8_t*)s_Data.pointVertexBufferBase);
			s_Data.pointVertexBuffer->SetData(s_Data.pointVertexBufferBase, dataSize);
			s_Data.pointMaterial->Bind();
			s_RendererAPI->DrawPoints(s_Data.pointVertexArray, s_Data.pointVertexCount);
		}

		// Lines
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