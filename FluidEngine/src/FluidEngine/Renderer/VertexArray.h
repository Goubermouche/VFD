#ifndef VERTEX_ARRAY_H_
#define VERTEX_ARRAY_H_

#include "FluidEngine/Renderer/Buffers/VertexBuffer.h"
#include "FluidEngine/Renderer/Buffers/IndexBuffer.h"

namespace fe {
	class VertexArray : public RefCounted
	{
	public:
		virtual ~VertexArray() {}

		virtual void Bind() const = 0;
		virtual void Unbind() const = 0;

		virtual void AddVertexBuffer(const Ref<VertexBuffer>& vertexBuffer) = 0;
		virtual void SetIndexBuffer(const Ref<IndexBuffer>& indexBuffer) = 0;

		virtual const std::vector<Ref<VertexBuffer>>& GetVertexBuffers() const = 0;
		virtual const Ref<IndexBuffer>& GetIndexBuffer() const = 0;

		static Ref<VertexArray> Create();
	};
}

#endif // !VERTEX_ARRAY_H_


