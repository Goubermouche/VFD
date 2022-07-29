#include "pch.h"
#include "Renderer.h"

#include <Glad/glad.h>

namespace fe {
	ShaderLibrary Renderer::shaderLibrary;
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
		shaderLibrary.AddShader("res/Shaders/Batched/PointShaderDiffuse.glsl");
		shaderLibrary.AddShader("res/Shaders/Batched/ColorShader.glsl");
		shaderLibrary.AddShader("res/Shaders/Normal/BasicDiffuseShader.glsl");
		shaderLibrary.AddShader("res/Shaders/Normal/PointDiffuseShader.glsl");

		// Initialize the batch renderer
		// Points
		s_Data.pointVertexArray = Ref<VertexArray>::Create(); 
		s_Data.pointVertexBuffer = Ref<VertexBuffer>::Create(s_Data.maxVertices * sizeof(PointVertex));
		s_Data.pointVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    },
			{ ShaderDataType::Float,  "a_Radius"   }
		});
		s_Data.pointVertexArray->AddVertexBuffer(s_Data.pointVertexBuffer);
		s_Data.pointVertexBufferBase = new PointVertex[s_Data.maxVertices];
		s_Data.pointMaterial = Ref<Material>::Create(shaderLibrary.GetShader("res/Shaders/Batched/PointShaderDiffuse.glsl"));

		// Lines
		s_Data.lineVertexArray = Ref<VertexArray>::Create();
		s_Data.lineVertexBuffer = Ref<VertexBuffer>::Create(s_Data.maxVertices * sizeof(LineVertex));
		s_Data.lineVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    }
		});
		s_Data.lineVertexArray->AddVertexBuffer(s_Data.lineVertexBuffer);
		s_Data.lineVertexBufferBase = new LineVertex[s_Data.maxVertices];
		s_Data.lineMaterial = Ref<Material>::Create(shaderLibrary.GetShader("res/Shaders/Batched/ColorShader.glsl"));

		// Quads
		s_Data.quadVertexArray = Ref<VertexArray>::Create();
		s_Data.quadVertexBuffer = Ref<VertexBuffer>::Create(s_Data.maxVertices * sizeof(QuadVertex));
		s_Data.quadVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    }
		});
		s_Data.quadVertexArray->AddVertexBuffer(s_Data.quadVertexBuffer);

		s_Data.quadVertexPositions[0] = { -0.5f, -0.5f, 0.0f, 1.0f };
		s_Data.quadVertexPositions[1] = {  0.5f, -0.5f, 0.0f, 1.0f };
		s_Data.quadVertexPositions[2] = {  0.5f,  0.5f, 0.0f, 1.0f };
		s_Data.quadVertexPositions[3] = { -0.5f,  0.5f, 0.0f, 1.0f };

		uint32_t* quadIndices = new uint32_t[s_Data.maxIndices];
		uint32_t quadIndexOffset = 0;

		for (uint32_t i = 0; i < s_Data.maxIndices; i += 6)
		{
			quadIndices[i + 0] = quadIndexOffset + 0;
			quadIndices[i + 1] = quadIndexOffset + 1;
			quadIndices[i + 2] = quadIndexOffset + 2;
			quadIndices[i + 3] = quadIndexOffset + 2;
			quadIndices[i + 4] = quadIndexOffset + 3;
			quadIndices[i + 5] = quadIndexOffset + 0;

			quadIndexOffset += 4;
		}

		Ref<IndexBuffer> quadIndexBuffer = Ref<IndexBuffer>::Create(quadIndices, s_Data.maxIndices);
		s_Data.quadVertexArray->SetIndexBuffer(quadIndexBuffer);

		s_Data.quadVertexBufferBase = new QuadVertex[s_Data.maxVertices];
		s_Data.quadMaterial = Ref<Material>::Create(shaderLibrary.GetShader("res/Shaders/Batched/ColorShader.glsl"));

		// Cubes
		s_Data.cubeVertexArray = Ref<VertexArray>::Create();

		s_Data.cubeVertexBuffer = Ref<VertexBuffer>::Create(s_Data.maxVertices * sizeof(CubeVertex));
		s_Data.cubeVertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" },
			{ ShaderDataType::Float4, "a_Color"    }
		});
		s_Data.cubeVertexArray->AddVertexBuffer(s_Data.cubeVertexBuffer);

		s_Data.cubeVertexPositions[0] = { -0.5f, -0.5f, -0.5f, 1.0f };
		s_Data.cubeVertexPositions[1] = {  0.5f, -0.5f, -0.5f, 1.0f };
		s_Data.cubeVertexPositions[2] = {  0.5f, -0.5f,  0.5f, 1.0f };
		s_Data.cubeVertexPositions[3] = { -0.5f, -0.5f,  0.5f, 1.0f };
		s_Data.cubeVertexPositions[4] = { -0.5f,  0.5f, -0.5f, 1.0f };
		s_Data.cubeVertexPositions[5] = {  0.5f,  0.5f, -0.5f, 1.0f };
		s_Data.cubeVertexPositions[6] = {  0.5f,  0.5f,  0.5f, 1.0f };
		s_Data.cubeVertexPositions[7] = { -0.5f,  0.5f,  0.5f, 1.0f };
		
		uint32_t* cubeIndices = new uint32_t[s_Data.maxIndices];
		uint32_t cubeIndexOffset = 0;

		for (uint32_t i = 0; i < s_Data.maxIndices; i += 24) {
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

		Ref<IndexBuffer> cubeIndexBuffer = Ref<IndexBuffer>::Create(cubeIndices, s_Data.maxIndices);
		s_Data.cubeVertexArray->SetIndexBuffer(cubeIndexBuffer);

		s_Data.cubeVertexBufferBase = new CubeVertex[s_Data.maxVertices];
		s_Data.cubeMaterial = Ref<Material>::Create(shaderLibrary.GetShader("res/Shaders/Batched/ColorShader.glsl"));

		LOG("renderer initialized successfully", "renderer", ConsoleColor::Purple);
	}

	void Renderer::BeginScene(Ref<Camera> camera)
	{
		s_Camera = camera;

		StartBatch();
	}

	void Renderer::EndScene()
	{
		Flush();
	}

	void Renderer::SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
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

	void Renderer::DrawPoint(const glm::vec3& p, const glm::vec4 color, float radius)
	{
		if (s_Data.pointVertexCount >= RendererData::maxVertices) {
			NextBatch();
		}

		s_Data.pointVertexBufferPtr->position = p;
		s_Data.pointVertexBufferPtr->color = color;
		s_Data.pointVertexBufferPtr->radius = radius;
		s_Data.pointVertexBufferPtr++;

		s_Data.pointVertexCount++;
	}

	void Renderer::DrawPoints(const Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material)
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
		if (s_Data.pointVertexCount >= RendererData::maxVertices) {
			NextBatch();
		}

		s_Data.lineVertexBufferPtr->position = p0;
		s_Data.lineVertexBufferPtr->color = color;
		s_Data.lineVertexBufferPtr++;

		s_Data.lineVertexBufferPtr->position = p1;
		s_Data.lineVertexBufferPtr->color = color;
		s_Data.lineVertexBufferPtr++;

		s_Data.lineVertexCount += 2;
	}

	void Renderer::DrawLines(const Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());

		material->Bind();
		vertexArray->Bind();

		glDrawArrays(GL_LINES, 0, vertexCount);
	
	}

	void Renderer::DrawLinesIndexed(const Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());

		material->Bind();
		vertexArray->Bind();
		 
		glDrawElements(GL_LINES, vertexCount, GL_UNSIGNED_INT, nullptr);
	}

	void Renderer::DrawQuad(const glm::mat4& transform, const glm::vec4& color) {
		if (s_Data.quadIndexCount >= RendererData::maxIndices) {
			NextBatch();
		}

		for (size_t i = 0; i < 4; i++)
		{
			s_Data.quadVertexBufferPtr->position = transform * s_Data.quadVertexPositions[i];
			s_Data.quadVertexBufferPtr->color = color;
			s_Data.quadVertexBufferPtr++;
		}

		s_Data.quadIndexCount += 6;
	}

	void Renderer::DrawBox(const glm::mat4& transform, const glm::vec4& color)
	{
		if (s_Data.cubeIndexCount >= RendererData::maxIndices) {
			NextBatch();
		}

		for (size_t i = 0; i < 8; i++)
		{
			s_Data.cubeVertexBufferPtr->position = transform * s_Data.cubeVertexPositions[i];
			s_Data.cubeVertexBufferPtr->color = color;
			s_Data.cubeVertexBufferPtr++;
		}

		s_Data.cubeIndexCount += 24;
	}

	void Renderer::DrawTriangles(const Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());
		material->Bind();

		vertexArray->Bind();
		glDrawArrays(GL_TRIANGLES, 0, vertexCount);
	}

	void Renderer::DrawTrianglesIndexed(const Ref<VertexArray> vertexArray, size_t count, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());
		material->Bind();

		vertexArray->Bind();
		glDrawElements(GL_TRIANGLES, vertexArray->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);
	}

	float Renderer::GetLineWidth()
	{
		return s_Data.lineWidth;
	}

	void Renderer::SetLineWidth(float width)
	{
		s_Data.lineWidth = width;
		glLineWidth(width);
	}

	void Renderer::StartBatch()
	{
		// Points
		s_Data.pointVertexCount = 0;
		s_Data.pointVertexBufferPtr = s_Data.pointVertexBufferBase;

		// Lines
		s_Data.lineVertexCount = 0;
		s_Data.lineVertexBufferPtr = s_Data.lineVertexBufferBase;

		// Quads
		s_Data.quadIndexCount = 0;
		s_Data.quadVertexBufferPtr = s_Data.quadVertexBufferBase;

		// Cubes
		s_Data.cubeIndexCount = 0;
		s_Data.cubeVertexBufferPtr = s_Data.cubeVertexBufferBase;
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
			s_Data.pointVertexBuffer->SetData(0, dataSize, s_Data.pointVertexBufferBase);
			DrawPoints(s_Data.pointVertexArray, s_Data.pointVertexCount, s_Data.pointMaterial);
		}

		// Lines
		if (s_Data.lineVertexCount)
		{
			uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.lineVertexBufferPtr - (uint8_t*)s_Data.lineVertexBufferBase);
			s_Data.lineVertexBuffer->SetData(0, dataSize, s_Data.lineVertexBufferBase);
			s_Data.lineMaterial->Bind();
			DrawLines(s_Data.lineVertexArray, s_Data.lineVertexCount, s_Data.lineMaterial);
		}

		// Quads
		if (s_Data.quadIndexCount)
		{
			uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.quadVertexBufferPtr - (uint8_t*)s_Data.quadVertexBufferBase);
			s_Data.quadVertexBuffer->SetData(0, dataSize, s_Data.quadVertexBufferBase);
			DrawTrianglesIndexed(s_Data.quadVertexArray, s_Data.quadIndexCount, s_Data.quadMaterial);
		}

		// Cubes
		if (s_Data.cubeIndexCount)
		{
			uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.cubeVertexBufferPtr - (uint8_t*)s_Data.cubeVertexBufferBase);
			s_Data.cubeVertexBuffer->SetData(0, dataSize, s_Data.cubeVertexBufferBase);
			DrawLinesIndexed(s_Data.cubeVertexArray, s_Data.cubeIndexCount, s_Data.cubeMaterial);
		}
	}
}