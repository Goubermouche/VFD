#ifndef UNIFORM_BUFFER_H
#define UNIFORM_BUFFER_H

namespace vfd {
	class UniformBuffer : public RefCounted
	{
	public:
		UniformBuffer(uint32_t size, uint32_t binding);
		~UniformBuffer();

		void SetData(const void* data, uint32_t size, uint32_t offset = 0) const;

		uint32_t GetRendererID() const;
		uint32_t GetBinding() const;
	private:
		uint32_t m_RendererID = 0;
		uint32_t m_Binding = 0;
	};
}

#endif // !UNIFORM_BUFFER_H