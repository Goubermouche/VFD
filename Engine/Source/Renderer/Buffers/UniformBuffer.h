#ifndef UNIFORM_BUFFER_H
#define UNIFORM_BUFFER_H

namespace fe {
	class UniformBuffer : public RefCounted
	{
	public:
		UniformBuffer(uint32_t size, uint32_t binding);
		~UniformBuffer();

		void SetData(const void* data, uint32_t size, uint32_t offset = 0) const;
	private:
		uint32_t m_RendererID = 0;
		uint32_t m_Binding = 0;
	};
}

#endif // !UNIFORM_BUFFER_H


