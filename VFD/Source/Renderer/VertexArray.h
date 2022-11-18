#ifndef VERTEX_ARRAY_H
#define VERTEX_ARRAY_H

#include "Renderer/Buffers/VertexBuffer.h"
#include "Renderer/Buffers/IndexBuffer.h"

namespace vfd {
	class VertexArray : public RefCounted
	{
	public:
		VertexArray();
		~VertexArray();

		void Bind() const;
		static void Unbind();

		void AddVertexBuffer(const Ref<VertexBuffer>& vertexBuffer);
		void SetIndexBuffer(const Ref<IndexBuffer>& indexBuffer);

		const std::vector<Ref<VertexBuffer>>& GetVertexBuffers() const;
		const Ref<IndexBuffer>& GetIndexBuffer() const;
		uint32_t GetRendererID() const;
	private:
		uint32_t m_RendererID = 0;
		Ref<IndexBuffer> m_IndexBuffer;
		std::vector<Ref<VertexBuffer>> m_VertexBuffers;
	};
}

#endif // !VERTEX_ARRAY_H


