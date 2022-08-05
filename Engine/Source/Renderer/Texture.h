#ifndef TEXTURE_H
#define TEXTURE_H

namespace fe {
	// TODO: add a struct builder
	class Texture : public RefCounted
	{
	public:
		Texture(const std::string& filepath);

		void Bind() const;
		static void Unbind();

		uint32_t GetRendererID() const {
			return m_RendererID;
		}

		const glm::uvec2& GetSize() {
			return m_Size;
		}
	private:
		uint32_t m_RendererID = 0;
		glm::uvec2 m_Size;
	};
}

#endif // !TEXTURE_H