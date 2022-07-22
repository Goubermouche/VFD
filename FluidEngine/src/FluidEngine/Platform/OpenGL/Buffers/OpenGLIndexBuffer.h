#ifndef OPENGL_INDEX_BUFFER_H_
#define OPENGL_INDEX_BUFFER_H_

#include "FluidEngine/Renderer/Buffers/IndexBuffer.h"

namespace fe::opengl {
	class OpenGLIndexBuffer : public IndexBuffer
	{
	public:
		OpenGLIndexBuffer(std::vector<uint32_t>& indices);
		OpenGLIndexBuffer(uint32_t* indices, uint32_t count);
		virtual ~OpenGLIndexBuffer();

		virtual uint32_t GetCount() const {
			return m_Count;
		}

		virtual uint32_t GetRendererID() const override;

		virtual void Bind() const;
		virtual void Unbind() const;
	private:
		uint32_t m_RendererID;
		uint32_t m_Count;
	};
}

#endif // !OPENGL_INDEX_BUFFER_H_