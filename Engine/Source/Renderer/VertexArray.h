#ifndef VERTEX_ARRAY_H
#define VERTEX_ARRAY_H

#include "Renderer/Buffers/VertexBuffer.h"
#include "Renderer/Buffers/IndexBuffer.h"

namespace fe {
	class VertexArray : public RefCounted
	{
	public:
		VertexArray();
		virtual ~VertexArray();

		void Bind() const;
		static void Unbind();

		void AddVertexBuffer(const Ref<VertexBuffer>& vertexBuffer);
		void SetIndexBuffer(const Ref<IndexBuffer>& indexBuffer);

		const std::vector<Ref<VertexBuffer>>& GetVertexBuffers() const {
			return m_VertexBuffers;
		}

		const Ref<IndexBuffer>& GetIndexBuffer() const {
			return m_IndexBuffer;
		}

		uint32_t GetRendererID() const;
	private:
		uint32_t m_RendererID = 0;
		std::vector<Ref<VertexBuffer>> m_VertexBuffers;
		Ref<IndexBuffer> m_IndexBuffer;
	};
}

#endif // !VERTEX_ARRAY_H


