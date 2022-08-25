#include "pch.h"
#include "VertexBuffer.h"

#include <Glad/glad.h>

namespace fe {
	VertexBuffer::VertexBuffer(const uint32_t size)
	{
		glCreateBuffers(1, &m_RendererID);
		Bind();
		glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	}

	VertexBuffer::VertexBuffer(uint32_t size, const void* data)
	{
		glCreateBuffers(1, &m_RendererID);
		Bind();
		glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
	}

	VertexBuffer::VertexBuffer(const std::vector<float>& vertices)
	{
		glCreateBuffers(1, &m_RendererID);
		Bind();
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
	}

	VertexBuffer::~VertexBuffer()
	{
		glDeleteBuffers(1, &m_RendererID);
	}

	void VertexBuffer::SetData(const int start, const uint32_t size, const void* data) const
	{
		Bind();
		glBufferSubData(GL_ARRAY_BUFFER, start, size, data);
	}

	void VertexBuffer::Bind() const
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
	}

	void VertexBuffer::Unbind()
	{
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}