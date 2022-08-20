#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>

namespace fe {
	enum class TextureFormat {
		None = 0,
		RGBA8,    // color
		RedInt,   // color + custom value (used for object picking & sampling the frame buffer)
		Depth,    // default
	};

	enum class TextureParameterName {
		None        = 0,
		MinFilter   = 0x2801,
		MagFilter   = 0x2800,
		WrapR       = 0x8072,
		WrapS       = 0x2802,
		WrapT       = 0x2803,
		CompareMode = GL_TEXTURE_COMPARE_MODE,
		CompareFunc = GL_TEXTURE_COMPARE_FUNC
	};				      

	enum class TextureTarget
	{
		Texture2D            = 0x0DE1,
		Texture2dMultiSample = 0x9100
	};

	enum class TextureParameterValue {
		None        = GL_NONE,
		// MinFilter, MagFilter
		Nearest     = 0x2600, 
		Linear      = 0x2601,
		// WrapR, WrapS, WrapT
		ClampToEdge = 0x812F,
		Repeat      = 0x2901,
		// Compare func
		LessOrEqual = GL_LEQUAL
	};

	struct TextureParameter {
		TextureParameterName Name = TextureParameterName::None;
		TextureParameterValue Value = TextureParameterValue::None;
	};

	struct TextureDescription {
		uint32_t Samples = 1;
		uint32_t Width = 0;
		uint32_t Height = 0;
		TextureFormat Format = TextureFormat::RGBA8;

		std::vector<TextureParameter> Parameters = {
			{ TextureParameterName::MinFilter, TextureParameterValue::Nearest }
		};
	};

	class Texture : public RefCounted
	{
	public:
		Texture(TextureDescription description);
		Texture(TextureDescription description, const std::string& filepath);
		~Texture();

		void Bind() const;
		void Unbind() const;

		uint32_t GetRendererID() const {
			return m_RendererID;
		}

		const uint32_t& GetWidth() const {
			return m_Description.Width;
		}

		const uint32_t& GetHeight() const {
			return m_Description.Height;
		}

		TextureTarget GetTarget() const;
	private:
		void Init(const unsigned char* data = nullptr);
		void Attach(GLenum internalFormat, GLenum format, const unsigned char* data = nullptr);
	private:
		TextureDescription m_Description;
		uint32_t m_RendererID = 0;
	};
}

#endif // !TEXTURE_H