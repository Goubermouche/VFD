#ifndef OPENGL_RENDERER_H_
#define OPENGL_RENDERER_H_

#include "FluidEngine/Renderer/RendererAPI.h"

namespace fe::opengl {
	class OpenGLRenderer : public RendererAPI
	{
	public:
		virtual void Init() override;

		virtual void Clear() override;

		virtual void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;
		virtual void SetClearColor(const glm::vec4& color) override;
		virtual void SetLineWidth(float lineWidth) override;

		virtual void DrawTriangles(const Ref<VertexArray> vertexArray, uint32_t vertexCount) override;
		virtual void DrawTrianglesIndexed(const Ref<VertexArray> vertexArray) override;
		virtual void DrawTrianglesIndexed(const Ref<VertexArray> vertexArray, uint32_t count) override;

		virtual void DrawLines(const Ref<VertexArray> vertexArray, uint32_t vertexCount) override;
		virtual void DrawLinesIndexed(const Ref<VertexArray> vertexArray, uint32_t count) override;

		virtual void DrawPoints(const Ref<VertexArray> vertexArray, uint32_t vertexCount) override;
	};
}

#endif // !OPENGL_RENDERER_H_
