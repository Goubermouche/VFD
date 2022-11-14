#include "pch.h"
#include "IndexBuffer.h"

#include <Glad/glad.h>

namespace fe {
	IndexBuffer::IndexBuffer(const std::vector<uint32_t>& indices)
		: m_Count(indices.size())
	{
		glCreateBuffers(1, &m_RendererID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);
	}

	IndexBuffer::IndexBuffer(const uint32_t* indices, const uint32_t count)
		: m_Count(count)
	{
		glCreateBuffers(1, &m_RendererID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(uint32_t), indices, GL_STATIC_DRAW);
	}

	IndexBuffer::~IndexBuffer()
	{
		glDeleteBuffers(1, &m_RendererID);
	}

	uint32_t IndexBuffer::GetCount() const
	{
		return m_Count;
	}

	uint32_t IndexBuffer::GetRendererID() const
	{
		return m_RendererID;
	};

	void IndexBuffer::Bind() const
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID);
	}

	void IndexBuffer::Unbind()
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
}