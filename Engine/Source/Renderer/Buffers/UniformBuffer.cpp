#include "pch.h"
#include "UniformBuffer.h"

#include <Glad/glad.h>

namespace fe {
	UniformBuffer::UniformBuffer(const uint32_t size, const uint32_t binding)
	{
		m_Binding = binding;

		glCreateBuffers(1, &m_RendererID);
		glNamedBufferData(m_RendererID, size, nullptr, GL_DYNAMIC_DRAW); // TODO: investigate usage hint
		glBindBufferBase(GL_UNIFORM_BUFFER, binding, m_RendererID);
	}

	UniformBuffer::~UniformBuffer()
	{
		glDeleteBuffers(1, &m_RendererID);
	}

	void UniformBuffer::SetData(const void* data, const uint32_t size, const uint32_t offset) const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, m_Binding,  m_RendererID);
		glNamedBufferSubData(m_RendererID, offset, size, data);
	}

	uint32_t UniformBuffer::GetRendererID() const
	{
		return m_RendererID;
	}

	uint32_t UniformBuffer::GetBinding() const
	{
		return m_Binding;
	}
}