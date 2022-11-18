#ifndef VERTEX_BUFFER_H
#define VERTEX_BUFFER_H

#include "Renderer/Shader.h"

namespace vfd {
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
		BufferElement() = default;
		BufferElement(ShaderDataType type, const std::string& name, const bool normalized = false);
			
		[[nodiscard]]
		uint32_t GetComponentCount() const;

		std::string Name;
		ShaderDataType Type = ShaderDataType::None;
		uint32_t Offset = 0;
		uint32_t Size = 0;
		bool Normalized = false;
	};

	class BufferLayout {
	public:
		BufferLayout() = default;
		BufferLayout(const std::initializer_list<BufferElement>& elements);
			
		uint32_t GetStride() const;

		const std::vector<BufferElement>& GetElements() const;
		
		std::vector<BufferElement>::iterator begin();
		std::vector<BufferElement>::iterator end();
		std::vector<BufferElement>::const_iterator begin() const;
		std::vector<BufferElement>::const_iterator end() const;
	private:
		void CalculateOffsetsAndStride();
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
		VertexBuffer(uint32_t size, const void* data);
		VertexBuffer(const std::vector<float>& vertices);
		~VertexBuffer();

		const BufferLayout& GetLayout() const;
		uint32_t GetRendererID() const;

		void SetLayout(const BufferLayout& layout);
		void SetData(int start, uint32_t size, const void* data) const;

		void Bind() const;
		static void Unbind();
	private:
		uint32_t m_RendererID = 0;
		BufferLayout m_Layout;
	};
}

#endif // !VERTEX_BUFFER_H
