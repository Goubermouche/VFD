#include "pch.h"
#include "Renderer.h"

#include <Glad/glad.h>

namespace fe {
	ShaderLibrary Renderer::s_ShaderLibrary;
	RendererData Renderer::s_Data = RendererData();
	Ref<Camera> Renderer::s_Camera = nullptr;

	void Renderer::Init()
	{
		// Initialize OpenGL
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glDepthFunc(GL_LESS);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		// Initialize shaders

		s_ShaderLibrary.AddShader("Resources/Shaders/Batched/PointShaderDiffuse.glsl");
		s_ShaderLibrary.AddShader("Resources/Shaders/Batched/ColorShader.glsl");
		s_ShaderLibrary.AddShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl");
		s_ShaderLibrary.AddShader("Resources/Shaders/Normal/PointDiffuseShader.glsl");

		// Initialize the batch renderer
		// Points
		s_Data.PointVertexArray = Ref<VertexArray>::Create(); 
		s_Data.PointVertexBuffer = Ref<VertexBuffer>::Create(s_Data.MaxVertices * sizeof(PointVertex));
		s_Data.PointVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    },
			{ ShaderDataType::Float,  "a_Radius"   }
		});
		s_Data.PointVertexArray->AddVertexBuffer(s_Data.PointVertexBuffer);
		s_Data.PointVertexBufferBase = new PointVertex[s_Data.MaxVertices];
		s_Data.PointMaterial = Ref<Material>::Create(GetShader("Resources/Shaders/Batched/PointShaderDiffuse.glsl"));

		// Lines
		s_Data.LineVertexArray = Ref<VertexArray>::Create();
		s_Data.LineVertexBuffer = Ref<VertexBuffer>::Create(s_Data.MaxVertices * sizeof(LineVertex));
		s_Data.LineVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    }
		});
		s_Data.LineVertexArray->AddVertexBuffer(s_Data.LineVertexBuffer);
		s_Data.LineVertexBufferBase = new LineVertex[s_Data.MaxVertices];
		s_Data.LineMaterial = Ref<Material>::Create(GetShader("Resources/Shaders/Batched/ColorShader.glsl"));

		// Quads
		s_Data.QuadVertexArray = Ref<VertexArray>::Create();
		s_Data.QuadVertexBuffer = Ref<VertexBuffer>::Create(s_Data.MaxVertices * sizeof(QuadVertex));
		s_Data.QuadVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    }
		});
		s_Data.QuadVertexArray->AddVertexBuffer(s_Data.QuadVertexBuffer);

		s_Data.QuadVertexPositions[0] = { -0.5f, -0.5f, 0.0f, 1.0f };
		s_Data.QuadVertexPositions[1] = {  0.5f, -0.5f, 0.0f, 1.0f };
		s_Data.QuadVertexPositions[2] = {  0.5f,  0.5f, 0.0f, 1.0f };
		s_Data.QuadVertexPositions[3] = { -0.5f,  0.5f, 0.0f, 1.0f };

		uint32_t* quadIndices = new uint32_t[s_Data.MaxIndices];
		uint32_t quadIndexOffset = 0;

		for (uint32_t i = 0; i < s_Data.MaxIndices; i += 6)
		{
			quadIndices[i + 0] = quadIndexOffset + 0;
			quadIndices[i + 1] = quadIndexOffset + 1;
			quadIndices[i + 2] = quadIndexOffset + 2;
			quadIndices[i + 3] = quadIndexOffset + 2;
			quadIndices[i + 4] = quadIndexOffset + 3;
			quadIndices[i + 5] = quadIndexOffset + 0;

			quadIndexOffset += 4;
		}

		const Ref<IndexBuffer> quadIndexBuffer = Ref<IndexBuffer>::Create(quadIndices, s_Data.MaxIndices);
		s_Data.QuadVertexArray->SetIndexBuffer(quadIndexBuffer);

		s_Data.QuadVertexBufferBase = new QuadVertex[s_Data.MaxVertices];
		s_Data.QuadMaterial = Ref<Material>::Create(GetShader("Resources/Shaders/Batched/ColorShader.glsl"));

		// Cubes
		s_Data.CubeVertexArray = Ref<VertexArray>::Create();

		s_Data.CubeVertexBuffer = Ref<VertexBuffer>::Create(s_Data.MaxVertices * sizeof(CubeVertex));
		s_Data.CubeVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    }
		});
		s_Data.CubeVertexArray->AddVertexBuffer(s_Data.CubeVertexBuffer);

		s_Data.CubeVertexPositions[0] = { -0.5f, -0.5f, -0.5f, 1.0f };
		s_Data.CubeVertexPositions[1] = {  0.5f, -0.5f, -0.5f, 1.0f };
		s_Data.CubeVertexPositions[2] = {  0.5f, -0.5f,  0.5f, 1.0f };
		s_Data.CubeVertexPositions[3] = { -0.5f, -0.5f,  0.5f, 1.0f };
		s_Data.CubeVertexPositions[4] = { -0.5f,  0.5f, -0.5f, 1.0f };
		s_Data.CubeVertexPositions[5] = {  0.5f,  0.5f, -0.5f, 1.0f };
		s_Data.CubeVertexPositions[6] = {  0.5f,  0.5f,  0.5f, 1.0f };
		s_Data.CubeVertexPositions[7] = { -0.5f,  0.5f,  0.5f, 1.0f };
		
		uint32_t* cubeIndices = new uint32_t[s_Data.MaxIndices];
		uint32_t cubeIndexOffset = 0;

		for (uint32_t i = 0; i < s_Data.MaxIndices; i += 24) {
			cubeIndices[i + 0]  = cubeIndexOffset + 0;
			cubeIndices[i + 1]  = cubeIndexOffset + 1;
			cubeIndices[i + 2]  = cubeIndexOffset + 1;
			cubeIndices[i + 3]  = cubeIndexOffset + 2;
			cubeIndices[i + 4]  = cubeIndexOffset + 2;
			cubeIndices[i + 5]  = cubeIndexOffset + 3;

			cubeIndices[i + 6]  = cubeIndexOffset + 3;
			cubeIndices[i + 7]  = cubeIndexOffset + 0;
			cubeIndices[i + 8]  = cubeIndexOffset + 4;
			cubeIndices[i + 9 ] = cubeIndexOffset + 5;
			cubeIndices[i + 10] = cubeIndexOffset + 5;
			cubeIndices[i + 11] = cubeIndexOffset + 6;

			cubeIndices[i + 12] = cubeIndexOffset + 6;
			cubeIndices[i + 13] = cubeIndexOffset + 7;
			cubeIndices[i + 14] = cubeIndexOffset + 7;
			cubeIndices[i + 15] = cubeIndexOffset + 4;
			cubeIndices[i + 16] = cubeIndexOffset + 0;
			cubeIndices[i + 17] = cubeIndexOffset + 4;

			cubeIndices[i + 18] = cubeIndexOffset + 1;
			cubeIndices[i + 19] = cubeIndexOffset + 5;
			cubeIndices[i + 20] = cubeIndexOffset + 2;
			cubeIndices[i + 21] = cubeIndexOffset + 6;
			cubeIndices[i + 22] = cubeIndexOffset + 3;
			cubeIndices[i + 23] = cubeIndexOffset + 7;

			cubeIndexOffset += 8;
		}

		const Ref<IndexBuffer> cubeIndexBuffer = Ref<IndexBuffer>::Create(cubeIndices, s_Data.MaxIndices);
		s_Data.CubeVertexArray->SetIndexBuffer(cubeIndexBuffer);

		s_Data.CubeVertexBufferBase = new CubeVertex[s_Data.MaxVertices];
		s_Data.CubeMaterial = Ref<Material>::Create(GetShader("Resources/Shaders/Batched/ColorShader.glsl"));

		LOG("renderer initialized successfully", "renderer", ConsoleColor::Purple);
	}

	void Renderer::BeginScene(const Ref<Camera> camera)
	{
		s_Camera = camera;

		StartBatch();
	}

	void Renderer::EndScene()
	{
		Flush();
	}

	void Renderer::SetViewport(const uint32_t x, const uint32_t y, const uint32_t width, const uint32_t height)
	{
		glViewport(x, y, width, height);
	}

	void Renderer::SetClearColor(const glm::vec4& color)
	{
		glClearColor(color.r, color.g, color.b, color.a);
	}

	void Renderer::Clear()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void Renderer::DrawPoint(const glm::vec3& p, const glm::vec4 color, const float radius)
	{
		if (s_Data.PointVertexCount >= RendererData::MaxVertices) {
			NextBatch();
		}

		s_Data.PointVertexBufferPtr->Position = p;
		s_Data.PointVertexBufferPtr->Color = color;
		s_Data.PointVertexBufferPtr->Radius = radius;
		s_Data.PointVertexBufferPtr++;

		s_Data.PointVertexCount++;
	}

	void Renderer::DrawPoints(const Ref<VertexArray> vertexArray, const size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());
		material->Set("viewportSize", s_Camera->GetViewportSize());

		material->Bind();
		vertexArray->Bind();

		glDrawArrays(GL_POINTS, 0, vertexCount);
	}

	void Renderer::DrawLine(const glm::vec3& p0, const glm::vec3& p1, const glm::vec4& color)
	{
		if (s_Data.PointVertexCount >= RendererData::MaxVertices) {
			NextBatch();
		}

		s_Data.LineVertexBufferPtr->Position = p0;
		s_Data.LineVertexBufferPtr->Color = color;
		s_Data.LineVertexBufferPtr++;

		s_Data.LineVertexBufferPtr->Position = p1;
		s_Data.LineVertexBufferPtr->Color = color;
		s_Data.LineVertexBufferPtr++;

		s_Data.LineVertexCount += 2;
	}

	void Renderer::DrawLines(const Ref<VertexArray> vertexArray, const size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());

		material->Bind();
		vertexArray->Bind();

		glDrawArrays(GL_LINES, 0, vertexCount);
	
	}

	void Renderer::DrawLinesIndexed(const Ref<VertexArray> vertexArray, const size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());

		material->Bind();
		vertexArray->Bind();
		 
		glDrawElements(GL_LINES, vertexCount, GL_UNSIGNED_INT, nullptr);
	}

	void Renderer::DrawQuad(const glm::mat4& transform, const glm::vec4& color) {
		if (s_Data.QuadIndexCount >= RendererData::MaxIndices) {
			NextBatch();
		}

		for (size_t i = 0; i < 4; i++)
		{
			s_Data.QuadVertexBufferPtr->Position = transform * s_Data.QuadVertexPositions[i];
			s_Data.QuadVertexBufferPtr->Color = color;
			s_Data.QuadVertexBufferPtr++;
		}

		s_Data.QuadIndexCount += 6;
	}

	void Renderer::DrawBox(const glm::mat4& transform, const glm::vec4& color)
	{
		if (s_Data.CubeIndexCount >= RendererData::MaxIndices) {
			NextBatch();
		}

		for (size_t i = 0; i < 8; i++)
		{
			s_Data.CubeVertexBufferPtr->Position = transform * s_Data.CubeVertexPositions[i];
			s_Data.CubeVertexBufferPtr->Color = color;
			s_Data.CubeVertexBufferPtr++;
		}

		s_Data.CubeIndexCount += 24;
	}

	void Renderer::DrawTriangles(const Ref<VertexArray> vertexArray, const size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());
		material->Bind();

		vertexArray->Bind();
		glDrawArrays(GL_TRIANGLES, 0, vertexCount);
	}

	void Renderer::DrawTrianglesIndexed(const Ref<VertexArray> vertexArray, const size_t count, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());
		material->Bind();

		vertexArray->Bind();
		glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr);
	}

	float Renderer::GetLineWidth()
	{
		return s_Data.lineWidth;
	}

	void Renderer::SetLineWidth(const float width)
	{
		s_Data.lineWidth = width;
		glLineWidth(width);
	}

	void Renderer::StartBatch()
	{
		// Points
		s_Data.PointVertexCount = 0;
		s_Data.PointVertexBufferPtr = s_Data.PointVertexBufferBase;

		// Lines
		s_Data.LineVertexCount = 0;
		s_Data.LineVertexBufferPtr = s_Data.LineVertexBufferBase;

		// Quads
		s_Data.QuadIndexCount = 0;
		s_Data.QuadVertexBufferPtr = s_Data.QuadVertexBufferBase;

		// Cubes
		s_Data.CubeIndexCount = 0;
		s_Data.CubeVertexBufferPtr = s_Data.CubeVertexBufferBase;
	}

	void Renderer::NextBatch()
	{
		Flush();
		StartBatch();
	}

	void Renderer::Flush()
	{
		// Points
		if (s_Data.PointVertexCount)
		{
			const uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.PointVertexBufferPtr - (uint8_t*)s_Data.PointVertexBufferBase);
			s_Data.PointVertexBuffer->SetData(0, dataSize, s_Data.PointVertexBufferBase);
			DrawPoints(s_Data.PointVertexArray, s_Data.PointVertexCount, s_Data.PointMaterial);
		}

		// Lines
		if (s_Data.LineVertexCount)
		{
			const uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.LineVertexBufferPtr - (uint8_t*)s_Data.LineVertexBufferBase);
			s_Data.LineVertexBuffer->SetData(0, dataSize, s_Data.LineVertexBufferBase);
			s_Data.LineMaterial->Bind();
			DrawLines(s_Data.LineVertexArray, s_Data.LineVertexCount, s_Data.LineMaterial);
		}

		// Quads
		if (s_Data.QuadIndexCount)
		{
			const uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.QuadVertexBufferPtr - (uint8_t*)s_Data.QuadVertexBufferBase);
			s_Data.QuadVertexBuffer->SetData(0, dataSize, s_Data.QuadVertexBufferBase);
			DrawTrianglesIndexed(s_Data.QuadVertexArray, s_Data.QuadIndexCount, s_Data.QuadMaterial);
		}

		// Cubes
		if (s_Data.CubeIndexCount)
		{
			const uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.CubeVertexBufferPtr - (uint8_t*)s_Data.CubeVertexBufferBase);
			s_Data.CubeVertexBuffer->SetData(0, dataSize, s_Data.CubeVertexBufferBase);
			DrawLinesIndexed(s_Data.CubeVertexArray, s_Data.CubeIndexCount, s_Data.CubeMaterial);
		}
	}
}