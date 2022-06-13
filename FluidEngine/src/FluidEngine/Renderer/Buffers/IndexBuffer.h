#ifndef INDEX_BUFFER_H_
#define INDEX_BUFFER_H_

namespace fe {
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



