#include "pch.h"
#include "Renderer.h"

#include <Glad/glad.h>

namespace vfd {
	ShaderLibrary Renderer::s_ShaderLibrary;
	RendererData Renderer::s_Data = RendererData();
	Ref<Camera> Renderer::s_Camera = nullptr;

	void Renderer::Init()
	{

		std::cout << "\n**Renderer information\n"
			         "Available shaders: \n";

		// Initialize OpenGL
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glDepthFunc(GL_LESS);
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		// Initialize shaders
		// Batched
		s_ShaderLibrary.AddShader("Resources/Shaders/Batched/PointShaderDiffuse.glsl");
		s_ShaderLibrary.AddShader("Resources/Shaders/Batched/ColorShader.glsl");

		// Normal
		s_ShaderLibrary.AddShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl");
		s_ShaderLibrary.AddShader("Resources/Shaders/Normal/PointDiffuseShader.glsl");
		s_ShaderLibrary.AddShader("Resources/Shaders/Normal/GridPlaneShader.glsl");
		
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

		// Cubes (wireframe)
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

		// Cubes (filled)
		//s_Data.CubeFilledVertexArray = Ref<VertexArray>::Create();

		//s_Data.CubeFilledVertexBuffer = Ref<VertexBuffer>::Create(s_Data.MaxVertices * sizeof(CubeVertex));
		//s_Data.CubeFilledVertexBuffer->SetLayout({
		//	{ ShaderDataType::Float3, "a_Position" }
		//});
		//s_Data.CubeFilledVertexArray->AddVertexBuffer(s_Data.CubeFilledVertexBuffer);

		//s_Data.CubeFilledVertexPositions[0] = { -0.5, -0.5,  0.5, 1.0f };
		//s_Data.CubeFilledVertexPositions[1] = {  0.5, -0.5,  0.5, 1.0f };
		//s_Data.CubeFilledVertexPositions[2] = {  0.5,  0.5,  0.5, 1.0f };
		//s_Data.CubeFilledVertexPositions[3] = { -0.5,  0.5,  0.5, 1.0f };
		//s_Data.CubeFilledVertexPositions[4] = { -0.5, -0.5, -0.5, 1.0f };
		//s_Data.CubeFilledVertexPositions[5] = {  0.5, -0.5, -0.5, 1.0f };
		//s_Data.CubeFilledVertexPositions[6] = {  0.5,  0.5, -0.5, 1.0f };
		//s_Data.CubeFilledVertexPositions[7] = { -0.5,  0.5, -0.5, 1.0f };

		//uint32_t* cubeFilledIndices = new uint32_t[s_Data.MaxIndices];
		//uint32_t cubeFilledIndexOffset = 0;

		//for (uint32_t i = 0; i < s_Data.MaxIndices; i += 36) {
		//	cubeFilledIndices[i + 0] = cubeFilledIndexOffset + 0;
		//	cubeFilledIndices[i + 1] = cubeFilledIndexOffset + 1;
		//	cubeFilledIndices[i + 2] = cubeFilledIndexOffset + 2;
		//	cubeFilledIndices[i + 3] = cubeFilledIndexOffset + 2;
		//	cubeFilledIndices[i + 4] = cubeFilledIndexOffset + 3;
		//	cubeFilledIndices[i + 5] = cubeFilledIndexOffset + 0;

		//	cubeFilledIndices[i + 6] = cubeFilledIndexOffset + 1;
		//	cubeFilledIndices[i + 7] = cubeFilledIndexOffset + 5;
		//	cubeFilledIndices[i + 8] = cubeFilledIndexOffset + 6;
		//	cubeFilledIndices[i + 9] = cubeFilledIndexOffset + 6;
		//	cubeFilledIndices[i + 10] = cubeFilledIndexOffset + 2;
		//	cubeFilledIndices[i + 11] = cubeFilledIndexOffset + 1;

		//	cubeFilledIndices[i + 12] = cubeFilledIndexOffset + 7;
		//	cubeFilledIndices[i + 13] = cubeFilledIndexOffset + 6;
		//	cubeFilledIndices[i + 14] = cubeFilledIndexOffset + 5;
		//	cubeFilledIndices[i + 15] = cubeFilledIndexOffset + 5;
		//	cubeFilledIndices[i + 16] = cubeFilledIndexOffset + 4;
		//	cubeFilledIndices[i + 17] = cubeFilledIndexOffset + 7;

		//	cubeFilledIndices[i + 18] = cubeFilledIndexOffset + 4;
		//	cubeFilledIndices[i + 19] = cubeFilledIndexOffset + 0;
		//	cubeFilledIndices[i + 20] = cubeFilledIndexOffset + 3;
		//	cubeFilledIndices[i + 21] = cubeFilledIndexOffset + 3;
		//	cubeFilledIndices[i + 22] = cubeFilledIndexOffset + 7;
		//	cubeFilledIndices[i + 23] = cubeFilledIndexOffset + 4;

		//	cubeFilledIndices[i + 24] = cubeFilledIndexOffset + 4;
		//	cubeFilledIndices[i + 25] = cubeFilledIndexOffset + 5;
		//	cubeFilledIndices[i + 26] = cubeFilledIndexOffset + 1;
		//	cubeFilledIndices[i + 27] = cubeFilledIndexOffset + 1;
		//	cubeFilledIndices[i + 28] = cubeFilledIndexOffset + 0;
		//	cubeFilledIndices[i + 29] = cubeFilledIndexOffset + 4;

		//	cubeFilledIndices[i + 30] = cubeFilledIndexOffset + 3;
		//	cubeFilledIndices[i + 31] = cubeFilledIndexOffset + 2;
		//	cubeFilledIndices[i + 32] = cubeFilledIndexOffset + 6;
		//	cubeFilledIndices[i + 33] = cubeFilledIndexOffset + 6;
		//	cubeFilledIndices[i + 34] = cubeFilledIndexOffset + 7;
		//	cubeFilledIndices[i + 35] = cubeFilledIndexOffset + 3;

		//	cubeFilledIndexOffset += 8;
		//}

		//const Ref<IndexBuffer> cubeFilledIndexBuffer = Ref<IndexBuffer>::Create(cubeFilledIndices, s_Data.MaxIndices);
		//s_Data.CubeFilledVertexArray->SetIndexBuffer(cubeFilledIndexBuffer);

		//s_Data.CubeFilledVertexBufferBase = new CubeFilledVertex[s_Data.MaxVertices];
		//s_Data.CubeFilledMaterial = Ref<Material>::Create(GetShader("Resources/Shaders/Batched/EntityIDBoundingBoxShader.glsl"));

		delete[] quadIndices;
		delete[] cubeIndices;
		//delete[] cubeFilledIndices;

		std::cout << '\n';
		// LOG("renderer initialized successfully", "renderer", ConsoleColor::Purple);
	}

	void Renderer::ShutDown()
	{
		delete[] s_Data.PointVertexBufferBase;
		delete[] s_Data.LineVertexBufferBase;
		delete[] s_Data.QuadVertexBufferBase;
		delete[] s_Data.CubeVertexBufferBase;
	}

	void Renderer::BeginScene(const Ref<Camera> camera)
	{
		s_Data.DrawCallCount = 0;
		s_Data.VertexCount = 0;
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
		s_Data.DrawCallCount++;
		s_Data.VertexCount += vertexCount;
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
		s_Data.DrawCallCount++;
		s_Data.VertexCount += vertexCount;
	}

	void Renderer::DrawLinesIndexed(const Ref<VertexArray> vertexArray, const size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());

		material->Bind();
		vertexArray->Bind();
		 
		glDrawElements(GL_LINES, vertexCount, GL_UNSIGNED_INT, nullptr);
		s_Data.DrawCallCount++;
		s_Data.VertexCount += vertexCount;
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

	//void Renderer::DrawBoxFilled(const glm::mat4& transform, const glm::vec4& color, const uint32_t id)
	//{
	//	if (s_Data.CubeFilledIndexCount >= RendererData::MaxIndices) {
	//		NextBatch();
	//	}

	//	for (size_t i = 0; i < 8; i++)
	//	{
	//		s_Data.CubeFilledVertexBufferPtr->Position = transform * s_Data.CubeFilledVertexPositions[i];
	//		s_Data.CubeFilledVertexBufferPtr++;
	//	}

	//	s_Data.CubeFilledIndexCount += 36;
	//}

	void Renderer::DrawTriangles(const Ref<VertexArray> vertexArray, const size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());
		material->Bind();

		vertexArray->Bind();
		glDrawArrays(GL_TRIANGLES, 0, vertexCount);
		s_Data.DrawCallCount++;
		s_Data.VertexCount += vertexCount;
	}

	void Renderer::DrawTrianglesIndexed(const Ref<VertexArray> vertexArray, const size_t vertexCount, Ref<Material> material)
	{
		material->Set("view", s_Camera->GetViewMatrix());
		material->Set("proj", s_Camera->GetProjectionMatrix());
		material->Bind();

		vertexArray->Bind();
		glDrawElements(GL_TRIANGLES, vertexCount, GL_UNSIGNED_INT, nullptr);
		s_Data.DrawCallCount++;
		s_Data.VertexCount += vertexCount;
	}

	float Renderer::GetLineWidth()
	{
		return s_Data.lineWidth;
	}

	Ref<Shader> Renderer::GetShader(const std::string& filepath)
	{
		return s_ShaderLibrary.GetShader(filepath);
	}

	uint32_t Renderer::GetDrawCallCount()
	{
		return s_Data.DrawCallCount;
	}

	uint32_t Renderer::GetVertexCount()
	{
		return s_Data.VertexCount;
	}

	const ShaderLibrary& Renderer::GetShaderLibrary()
	{
		return s_ShaderLibrary;
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

		// Cubes (wireframe)
		s_Data.CubeIndexCount = 0;
		s_Data.CubeVertexBufferPtr = s_Data.CubeVertexBufferBase;

		// Cubes (filled)
		//s_Data.CubeFilledIndexCount = 0;
		//s_Data.CubeFilledVertexBufferPtr = s_Data.CubeFilledVertexBufferBase;
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

		// Cubes (wireframe)
		if (s_Data.CubeIndexCount)
		{
			const uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.CubeVertexBufferPtr - (uint8_t*)s_Data.CubeVertexBufferBase);
			s_Data.CubeVertexBuffer->SetData(0, dataSize, s_Data.CubeVertexBufferBase);
			DrawLinesIndexed(s_Data.CubeVertexArray, s_Data.CubeIndexCount, s_Data.CubeMaterial);
		}

		// Cubes (filled)
		//if (s_Data.CubeFilledIndexCount)
		//{
		//	const uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.CubeFilledVertexBufferPtr - (uint8_t*)s_Data.CubeFilledVertexBufferBase);
		//	s_Data.CubeFilledVertexBuffer->SetData(0, dataSize, s_Data.CubeFilledVertexBufferBase);
		//	DrawTrianglesIndexed(s_Data.CubeFilledVertexArray, s_Data.CubeFilledIndexCount, s_Data.CubeFilledMaterial);
		//}
	}
}