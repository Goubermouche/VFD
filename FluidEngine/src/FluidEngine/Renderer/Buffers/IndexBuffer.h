#ifndef INDEX_BUFFER_H_
#define INDEX_BUFFER_H_

namespace fe {
	/// <summary>
	/// Simple IBO class, stores indices that can be used inside a VAO to decrease the number of vertices by removing duplicates.
	/// </summary>
	class IndexBuffer : public RefCounted
	{
	public:
		virtual ~IndexBuffer() {}

		virtual void Bind() const = 0;
		virtual void Unbind() const = 0;

		virtual uint32_t GetCount() const = 0;

		static Ref<IndexBuffer> Create(std::vector<uint32_t>& indices);
	};
}

#endif // !INDEX_BUFFER_H_



