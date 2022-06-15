#ifndef RENDERER_H_
#define RENDERER_H_

#include "RendererAPI.h"

#include "FluidEngine/Renderer/VertexArray.h"
#include "FluidEngine/Renderer/Material.h"
#include "FluidEngine/Renderer/Buffers/FrameBuffer.h"

#include "FluidEngine/Editor/EditorCamera.h"

namespace fe {
	#pragma region Batch rendering
	struct LineVertex {
		glm::vec3 position;
		glm::vec4 color;
	};

	struct RendererData {
		static const uint32_t maxQuads = 20000;
		static const uint32_t maxVertices = maxQuads * 4;
		static const uint32_t maxIndices = maxQuads * 6;

		Ref<VertexArray> lineVertexArray;
		Ref<VertexBuffer> lineVertexBuffer;
		Ref<Material> lineMaterial;

		uint32_t lineVertexCount = 0;
		LineVertex* lineVertexBufferBase = nullptr;
		LineVertex* lineVertexBufferPtr = nullptr;
		float lineWidth = 2;
	};
	#pragma endregion

	class EditorCamera;

	/// <summary>
	/// Base renderer class. Enables us to interact with the current renderer API. 
	/// </summary>
	class Renderer
	{
	public:
		static void Init();

		static void BeginScene(Ref<EditorCamera> camera);
		static void EndScene();

		static void Clear();

		static void DrawLine(const glm::vec3& p0, const glm::vec3& p1, const glm::vec4& color);
		static void DrawBox(const glm::vec3& position, const glm::vec3& size, const glm::vec4& color);

		static void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);
		static void SetClearColor(const glm::vec4& color);
		static void SetLineWidth(float width);

		static float GetLineWidth();

		inline static RendererAPIType GetAPI() {
			return RendererAPI::GetAPI();
		}
	private:
		static void StartBatch();
		static void NextBatch();
		static void Flush();
	private:
		/// <summary>
		/// Current renderer API.
		/// </summary>
		static RendererAPI* s_RendererAPI;
		static RendererData s_Data;
		static Ref<EditorCamera> s_Camera;
	};
}

#endif // !RENDERER_H_