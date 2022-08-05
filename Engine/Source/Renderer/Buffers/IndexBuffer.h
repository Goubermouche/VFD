#ifndef INDEX_BUFFER_H
#define INDEX_BUFFER_H

namespace fe {
	/// <summary>
	/// Simple IBO class, stores indices that can be used inside a VAO to decrease the number of vertices by removing duplicates.
	/// </summary>
	class IndexBuffer : public RefCounted
	{
	public:
		IndexBuffer(const std::vector<uint32_t>& indices);
		IndexBuffer(const uint32_t* indices, uint32_t count);
		~IndexBuffer();

		uint32_t GetCount() const {
			return m_Count;
		}

		uint32_t GetRendererID() const
		{
			return m_RendererID;
		};

		void Bind() const;
		static void Unbind();
	private:
		uint32_t m_RendererID = 0;
		uint32_t m_Count;
	};
}

#endif // !INDEX_BUFFER_H