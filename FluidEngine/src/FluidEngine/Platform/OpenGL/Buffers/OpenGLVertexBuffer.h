#ifndef OPENGL_VERTEX_BUFFER_H_
#define OPENGL_VERTEX_BUFFER_H_

#include "FluidEngine/Renderer/Buffers/VertexBuffer.h"

namespace fe::opengl {
	class OpenGLVertexBuffer : public VertexBuffer
	{
	public:
		OpenGLVertexBuffer(uint32_t size);
		OpenGLVertexBuffer(std::vector<float>& vertices);
		virtual ~OpenGLVertexBuffer();

		virtual const BufferLayout& GetLayout() const override {
			return m_Layout;
		}
		virtual void SetLayout(const BufferLayout& layout) override {
			m_Layout = layout;
		}

		virtual void SetData(const void* data, uint32_t size) override;

		virtual void Bind() const override;
		virtual void Unbind() const override;
	private:
		uint32_t m_RendererID;
		BufferLayout m_Layout;
	};
}

#endif // !OPENGL_VERTEX_BUFFER_H_