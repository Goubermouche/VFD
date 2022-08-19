#ifndef RENDERER_H
#define RENDERER_H

#include "Renderer/VertexArray.h"
#include "Renderer/Material.h"
#include "Renderer/Camera.h"
#include "Renderer/Buffers/FrameBuffer.h"

namespace fe {
	struct PointVertex {
		glm::vec3 Position;
		glm::vec4 Color;
		float Radius = 0.0f;
	};

	struct LineVertex {
		glm::vec3 Position;
		glm::vec4 Color;
	};

	struct QuadVertex {
		glm::vec3 Position;
		glm::vec4 Color;
	};

	struct CubeVertex {
		glm::vec3 Position;
		glm::vec4 Color;
	};

	/// <summary>
	/// Batch renderer data
	/// </summary>
	struct RendererData {
		static constexpr  uint32_t MaxQuads = 20000;
		static constexpr uint32_t MaxVertices = MaxQuads * 4;
		static constexpr uint32_t MaxIndices = MaxQuads * 24;
		uint32_t DrawCallCount = 0;
		uint32_t VertexCount = 0;

		float lineWidth = 1;

		// Points
		Ref<VertexArray> PointVertexArray;
		Ref<VertexBuffer> PointVertexBuffer;
		Ref<Material> PointMaterial;

		uint32_t PointVertexCount = 0;
		PointVertex* PointVertexBufferBase = nullptr;
		PointVertex* PointVertexBufferPtr = nullptr;

		// Lines
		Ref<VertexArray> LineVertexArray;
		Ref<VertexBuffer> LineVertexBuffer;
		Ref<Material> LineMaterial;

		uint32_t LineVertexCount = 0;
		LineVertex* LineVertexBufferBase = nullptr;
		LineVertex* LineVertexBufferPtr = nullptr;

		// Quads
		Ref<VertexArray> QuadVertexArray;
		Ref<VertexBuffer> QuadVertexBuffer;
		Ref<Material> QuadMaterial;

		uint32_t QuadIndexCount = 0;
		QuadVertex* QuadVertexBufferBase = nullptr;
		QuadVertex* QuadVertexBufferPtr = nullptr;
		glm::vec4 QuadVertexPositions[4];

		// Cubes
		Ref<VertexArray> CubeVertexArray;
		Ref<VertexBuffer> CubeVertexBuffer;
		Ref<Material> CubeMaterial;

		uint32_t CubeIndexCount = 0;
		CubeVertex* CubeVertexBufferBase = nullptr;
		CubeVertex* CubeVertexBufferPtr = nullptr;
		glm::vec4 CubeVertexPositions[8];
	};

	class Camera;

	/// <summary>
	/// Base renderer class. Enables us to interact with the current renderer API. 
	/// </summary>
	class Renderer
	{
	public:
		static void Init();
		static void ShutDown();

		/// <summary>
		/// Starts a new render 'context' using the specified camera. All objects drawn in this context will use the specified camera for projection and view matrices. Additionally, the batch renderer is readied.
		/// </summary>
		/// <param name="camera"></param>
		static void BeginScene(Ref<Camera> camera);

		/// <summary>
		/// Ends the current render context.Submits the last batch to the render API.
		/// </summary>
		static void EndScene();

		/// <summary>
		/// Clears the viewport with the previously specified clear color. 
		/// </summary>
		static void Clear();

		/// <summary>
		/// Draws a point using the batch renderer.
		/// </summary>
		/// <param name="p">Point location.</param>
		/// <param name="color">Point color.</param>
		/// <param name="radius">Point radius.</param>
		static void DrawPoint(const glm::vec3& p, glm::vec4 color, float radius = 1.0f);
		static void DrawPoints(Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material);

		/// <summary>
		/// Draws a line using the batch renderer.
		/// </summary>
		/// <param name="p0">First point of the line.</param>
		/// <param name="p1">Second point of the line.</param>
		/// <param name="color">Color to drwa the line in.</param>
		static void DrawLine(const glm::vec3& p0, const glm::vec3& p1, const glm::vec4& color);
		static void DrawLines(Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material);
		static void DrawLinesIndexed(Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material);

		static void DrawQuad(const glm::mat4& transform, const glm::vec4& color);

		static void DrawBox(const glm::mat4& transform, const glm::vec4& color);

		static void DrawTriangles(Ref<VertexArray> vertexArray, size_t vertexCount, Ref<Material> material);
		static void DrawTrianglesIndexed(Ref<VertexArray> vertexArray, size_t count, Ref<Material> material);

		/// <summary>
		/// Sets the viewports position and size.
		/// </summary>
		/// <param name="x">Position on the X axis.</param>
		/// <param name="y">Position on the Y axis.</param>
		/// <param name="width">Viewport width.</param>
		/// <param name="height">Viewport height.</param>
		static void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

		/// <summary>
		/// Sets the new clear color that will be used by the Clear function until a new clear color is set.
		/// </summary>
		/// <param name="color">Clear color.</param>
		static void SetClearColor(const glm::vec4& color);

		/// <summary>
		/// Sets the line width that will be used by the batch renderer until a new line width is set. Can only be used once per context.
		/// </summary>
		/// <param name="width">Line width.</param>
		static void SetLineWidth(float width);

		/// <summary>
		/// Gets the line width currently used by the batch renderer.
		/// </summary>
		/// <returns>Currently used line width.</returns>
		static float GetLineWidth();

		static Ref<Shader> GetShader(const std::string& filepath)
		{
			return s_ShaderLibrary.GetShader(filepath);
		}

		// Stats
		static uint32_t GetDrawCallCount()
		{
			return s_Data.DrawCallCount;
		}

		static uint32_t GetVertexCount() {
			return s_Data.VertexCount;
		}
	private:
		// Batching
		static void StartBatch();
		static void NextBatch();
		static void Flush();
	private:
		/// <summary>
		/// Buffer of render data for the current batch.
		/// </summary>
		static RendererData s_Data;

		/// <summary>
		/// Camera that is currently used by the renderer, set by calling the BeginScene function
		/// </summary>
		static Ref<Camera> s_Camera;
		static ShaderLibrary s_ShaderLibrary;
	};
}

#endif // !RENDERER_H_