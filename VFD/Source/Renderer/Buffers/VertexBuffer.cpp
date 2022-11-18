#include "pch.h"
#include "VertexBuffer.h"

#include <Glad/glad.h>

namespace vfd {
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

	const BufferLayout& VertexBuffer::GetLayout() const
	{
		return m_Layout;
	}

	uint32_t VertexBuffer::GetRendererID() const
	{
		return m_RendererID;
	}

	void VertexBuffer::SetLayout(const BufferLayout& layout)
	{
		m_Layout = layout;
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

	BufferElement::BufferElement(ShaderDataType type, const std::string& name, const bool normalized)
		: Name(name), Type(type), Offset(0), Size(ShaderDataTypeSize(type)), Normalized(normalized) {
	}

	uint32_t BufferElement::GetComponentCount() const
	{
		switch (Type)
		{
		case ShaderDataType::Bool:   return 1;
		case ShaderDataType::Int:    return 1;
		case ShaderDataType::Uint:   return 1;
		case ShaderDataType::Float:  return 1;
		case ShaderDataType::Float2: return 2;
		case ShaderDataType::Float3: return 3;
		case ShaderDataType::Float4: return 4;
		case ShaderDataType::Mat3:   return 3 * 3;
		case ShaderDataType::Mat4:   return 4 * 4;
		}

		ERROR("unknown shader data type!");
		return 0;
	}

	BufferLayout::BufferLayout(const std::initializer_list<BufferElement>& elements)
		: m_Elements(elements)
	{
		CalculateOffsetsAndStride();
	}

	uint32_t BufferLayout::GetStride() const
	{
		return m_Stride;
	}

	const std::vector<BufferElement>& BufferLayout::GetElements() const
	{
		return m_Elements;
	}

	std::vector<BufferElement>::iterator BufferLayout::begin()
	{
		return m_Elements.begin();
	}

	std::vector<BufferElement>::iterator BufferLayout::end()
	{
		return m_Elements.end();
	}

	std::vector<BufferElement>::const_iterator BufferLayout::begin() const
	{
		return m_Elements.begin();
	}

	std::vector<BufferElement>::const_iterator BufferLayout::end() const
	{
		return m_Elements.end();
	}

	void BufferLayout::CalculateOffsetsAndStride()
	{
		uint32_t offset = 0;
		m_Stride = 0;

		for (auto& element : m_Elements) {
			element.Offset = offset;
			offset += element.Size;
			m_Stride += element.Size;
		}
	}
}