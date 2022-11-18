#ifndef INDEX_BUFFER_H
#define INDEX_BUFFER_H

namespace vfd {
	/// <summary>
	/// Simple IBO class, stores indices that can be used inside a VAO to decrease the number of vertices by removing duplicates.
	/// </summary>
	class IndexBuffer : public RefCounted
	{
	public:
		IndexBuffer(const std::vector<uint32_t>& indices);
		IndexBuffer(const uint32_t* indices, uint32_t count);
		~IndexBuffer();

		uint32_t GetCount() const;
		uint32_t GetRendererID() const;

		void Bind() const;
		static void Unbind();
	private:
		uint32_t m_RendererID = 0;
		uint32_t m_Count = 0;
	};
}

#endif // !INDEX_BUFFER_H