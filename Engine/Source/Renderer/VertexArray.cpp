#include "pch.h"
#include "VertexArray.h"

#include <Glad/glad.h>

namespace fe {
	GLenum ShaderDataTypeToOpenGLBaseType(const ShaderDataType type) {
		switch (type)
		{
		case ShaderDataType::Bool:   return GL_BOOL;
		case ShaderDataType::Int:    return GL_INT;
		case ShaderDataType::Uint:   return GL_UNSIGNED_INT;
		case ShaderDataType::Float:  return GL_FLOAT;
		case ShaderDataType::Float2: return GL_FLOAT;
		case ShaderDataType::Float3: return GL_FLOAT;
		case ShaderDataType::Float4: return GL_FLOAT;
		case ShaderDataType::Mat3:   return GL_FLOAT;
		case ShaderDataType::Mat4:   return GL_FLOAT;
		}

		ASSERT(false, "unknown ShaderDataType!");
		return 0;
	}

	VertexArray::VertexArray()
	{
		glCreateVertexArrays(1, &m_RendererID);
		Unbind();
	}

	VertexArray::~VertexArray()
	{
		glDeleteVertexArrays(1, &m_RendererID);
	}

	void VertexArray::Bind() const
	{
		glBindVertexArray(m_RendererID);
	}

	void VertexArray::Unbind()
	{
		glBindVertexArray(0);
	}

	void VertexArray::AddVertexBuffer(const Ref<VertexBuffer>& vertexBuffer)
	{
		if (vertexBuffer->GetLayout().GetElements().empty()) {
			ASSERT(false, "vertex buffer has no layout!");
			return;
		}

		glBindVertexArray(m_RendererID);
		vertexBuffer->Bind();

		uint32_t index = 0;
		const auto& layout = vertexBuffer->GetLayout();
		for (const auto& element : layout)
		{
			switch (element.Type)
			{
			case ShaderDataType::Float:
			case ShaderDataType::Float2:
			case ShaderDataType::Float3:
			case ShaderDataType::Float4:
			{
				glEnableVertexAttribArray(index);
				glVertexAttribPointer(index,
					element.GetComponentCount(),
					ShaderDataTypeToOpenGLBaseType(element.Type),
					element.Normalized ? GL_TRUE : GL_FALSE,
					layout.GetStride(),
					(const void*)element.Offset);
				index++;
				break;
			}
			case ShaderDataType::Int:
			case ShaderDataType::Uint:
			{

			}
			case ShaderDataType::Bool:
			{
				glEnableVertexAttribArray(index);
				glVertexAttribIPointer(index,
					element.GetComponentCount(),
					ShaderDataTypeToOpenGLBaseType(element.Type),
					layout.GetStride(),
					(const void*)element.Offset);
				index++;
				break;
			}
			case ShaderDataType::Mat3:
			case ShaderDataType::Mat4:
			{
				uint8_t count = element.GetComponentCount();
				for (uint8_t i = 0; i < count; i++)
				{
					glEnableVertexAttribArray(index);
					glVertexAttribPointer(index,
						count,
						ShaderDataTypeToOpenGLBaseType(element.Type),
						element.Normalized ? GL_TRUE : GL_FALSE,
						layout.GetStride(),
						(const void*)(element.Offset + sizeof(float) * count * i));
					glVertexAttribDivisor(index, 1);
					index++;
				}
				break;
			}
			default:
				ASSERT(false, "unknown ShaderDataType!");
			}
		}

		m_VertexBuffers.push_back(vertexBuffer);
	}

	void VertexArray::SetIndexBuffer(const Ref<IndexBuffer>& indexBuffer)
	{
		glBindVertexArray(m_RendererID);
		indexBuffer->Bind();
		m_IndexBuffer = indexBuffer;
	}

	const std::vector<Ref<VertexBuffer>>& VertexArray::GetVertexBuffers() const
	{
		return m_VertexBuffers;
	}

	const Ref<IndexBuffer>& VertexArray::GetIndexBuffer() const
	{
		return m_IndexBuffer;
	}

	uint32_t VertexArray::GetRendererID() const
	{
		return m_RendererID;
	}
}
