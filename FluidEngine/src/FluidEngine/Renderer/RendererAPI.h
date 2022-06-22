#ifndef RENDERER_API_H_
#define RENDERER_API_H_

#include "FluidEngine/Renderer/VertexArray.h"

namespace fe {

	/// <summary>
	/// Currently supported renderer API's
	/// </summary>
	enum class RendererAPIType {
		None,
		OpenGL
	};

	/// <summary>
	/// Renderer API, holds  level highimplementations of common graphics library functions.
	/// </summary>
	class RendererAPI
	{
	public:
		virtual void Init() = 0;

		virtual void Clear() = 0;

		virtual void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) = 0;
		virtual void SetClearColor(const glm::vec4& color) = 0;
		virtual void SetLineWidth(float lineWidth) = 0;

		virtual void DrawIndexed(const Ref<VertexArray> vertexArray) = 0;
		virtual void DrawPoints(const Ref<VertexArray> vertexArray, uint32_t vertexCount) = 0;
		virtual void DrawLines(const Ref<VertexArray> vertexArray, uint32_t vertexCount) = 0;

		static inline RendererAPIType GetAPIType() {
			return s_API;
		}

		static void SetAPI(RendererAPIType api);
	private:
		static RendererAPIType s_API;
	};
}

#endif // !RENDERER_API_H_