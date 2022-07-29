#ifndef INDEX_BUFFER_H_
#define INDEX_BUFFER_H_

namespace fe {
	/// <summary>
	/// Simple IBO class, stores indices that can be used inside a VAO to decrease the number of vertices by removing duplicates.
	/// </summary>
	class IndexBuffer : public RefCounted
	{
	public:
		IndexBuffer(std::vector<uint32_t>& indices);
		IndexBuffer(uint32_t* indices, uint32_t count);
		virtual ~IndexBuffer();

		uint32_t GetCount() const {
			return m_Count;
		}

		uint32_t GetRendererID() const;

		void Bind() const;
		void Unbind() const;
	private:
		uint32_t m_RendererID;
		uint32_t m_Count;
	};
}

#endif // !INDEX_BUFFER_H_