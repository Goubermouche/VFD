#ifndef VERTEX_BUFFER_H
#define VERTEX_BUFFER_H

#include "Renderer/Shader.h"

namespace fe {
	static uint32_t ShaderDataTypeSize(const ShaderDataType type) {
		switch (type)
		{
		case ShaderDataType::Bool:   return 1;
		case ShaderDataType::Int:    return 4;
		case ShaderDataType::Uint:   return 4;
		case ShaderDataType::Float:  return 4;
		case ShaderDataType::Float2: return 4 * 2;
		case ShaderDataType::Float3: return 4 * 3;
		case ShaderDataType::Float4: return 4 * 4;
		case ShaderDataType::Mat3:   return 4 * 3 * 3;
		case ShaderDataType::Mat4:   return 4 * 4 * 4;
		}

		ERROR("unknown shader data type!");
		return 0;
	}

	struct BufferElement {
		std::string name;
		ShaderDataType type;
		uint32_t offset;
		uint32_t size;
		bool normalized;

		BufferElement() = default;
		BufferElement(ShaderDataType type, const std::string& name,const bool normalized = false)
			: name(name), type(type), offset(0), size(ShaderDataTypeSize(type)), normalized(normalized) {
		}

		uint32_t GetComponentCount() const {
			switch (type)
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
	};

	class BufferLayout {
	public:
		BufferLayout() = default;
		BufferLayout(const std::initializer_list<BufferElement>& elements)
			: m_Elements(elements)
		{
			CalculateOffsetsAndStride();
		}

		uint32_t GetStride() const
		{
			return m_Stride;
		}

		const std::vector<BufferElement>& GetElements() const
		{
			return m_Elements;
		}

		std::vector<BufferElement>::iterator begin()
		{
			return m_Elements.begin();
		}

		std::vector<BufferElement>::iterator end()
		{
			return m_Elements.end();
		}

		[[nodiscard]]
		std::vector<BufferElement>::const_iterator begin() const
		{
			return m_Elements.begin();
		}

		[[nodiscard]]
		std::vector<BufferElement>::const_iterator end() const
		{
			return m_Elements.end();
		}
	private:
		void CalculateOffsetsAndStride() {
			uint32_t offset = 0;
			m_Stride = 0;

			for (auto& element : m_Elements) {
				element.offset = offset;
				offset += element.size;
				m_Stride += element.size;
			}
		}
	private:
		std::vector<BufferElement> m_Elements;
		uint32_t m_Stride = 0;
	};

	/// <summary>
	/// Simple VBO class, holds vertices used inside of a VAO.
	/// </summary>
	class VertexBuffer : public RefCounted {
	public:
		VertexBuffer(uint32_t size);
		VertexBuffer(const std::vector<float>& vertices);
		virtual ~VertexBuffer();

		const BufferLayout& GetLayout() const {
			return m_Layout;
		}

		uint32_t GetRendererID() {
			return m_RendererID;
		}

		void SetLayout(const BufferLayout& layout) {
			m_Layout = layout;
		}

		void SetData(int start, uint32_t size, const void* data) const;

		void Bind() const;
		static void Unbind();
	private:
		uint32_t m_RendererID = 0;
		BufferLayout m_Layout;
	};
}

#endif // !VERTEX_BUFFER_H
