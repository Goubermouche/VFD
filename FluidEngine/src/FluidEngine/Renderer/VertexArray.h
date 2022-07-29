#ifndef VERTEX_ARRAY_H_
#define VERTEX_ARRAY_H_

#include "FluidEngine/Renderer/Buffers/VertexBuffer.h"
#include "FluidEngine/Renderer/Buffers/IndexBuffer.h"

namespace fe {
	class VertexArray : public RefCounted
	{
	public:
		VertexArray();
		virtual ~VertexArray();

		void Bind() const;
		void Unbind() const;

		void AddVertexBuffer(const Ref<VertexBuffer>& vertexBuffer);
		void SetIndexBuffer(const Ref<IndexBuffer>& indexBuffer);

		const std::vector<Ref<VertexBuffer>>& GetVertexBuffers() const {
			return m_VertexBuffers;
		}

		const Ref<IndexBuffer>& GetIndexBuffer() const {
			return m_IndexBuffer;
		}

		uint32_t GetRendererID();
	private:
		uint32_t m_RendererID;
		std::vector<Ref<VertexBuffer>> m_VertexBuffers;
		Ref<IndexBuffer> m_IndexBuffer;
	};
}

#endif // !VERTEX_ARRAY_H_


