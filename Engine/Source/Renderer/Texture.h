#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>

namespace fe {
	enum class TextureFormat {
		None        = 0,
		RGBA8,                        // color
		RedInt,	                      // color + custom value (used for object picking & sampling the frame buffer)
		Depth24Stencil8,              // depth / stencil
		Depth       = Depth24Stencil8 // default
	};

	enum class TextureParameterName {
		None        = 0,
		MinFilter   = 0x2801,
		MagFilter   = 0x2800,
		WrapR       = 0x8072,
		WrapS       = 0x2802,
		WrapT       = 0x2803
	};

	enum class TextureParameterValue {
		None        = 0,
		// MinFilter, MagFilter
		Nearest     = 0x2600, 
		Linear      = 0x2601,
		// WrapR, WrapS, WrapT
		ClampToEdge = 0x812F,
		Repeat      = 0x2901
	};

	struct TextureParameter {
		TextureParameterName Name = TextureParameterName::None;
		TextureParameterValue Value = TextureParameterValue::None;
	};

	struct TextureDesc {
		uint32_t samples = 1;
		uint32_t Width = 0;
		uint32_t Height = 0;
		TextureFormat Format = TextureFormat::RGBA8;

		std::vector<TextureParameter> Parameters = {
			{ TextureParameterName::MinFilter, TextureParameterValue::Linear }
		};
	};

	// TODO: add a struct builder
	class Texture : public RefCounted
	{
	public:
		Texture(const TextureDesc& description);
		Texture(const TextureDesc& description, const std::string& filepath);

		void Bind() const;
		static void Unbind();

		uint32_t GetRendererID() const {
			return m_RendererID;
		}

		const uint32_t GetWidth() const {
			return m_Description.Width;
		}

		const uint32_t GetHeight() const {
			return m_Description.Height;
		}
	private:
		void Attach(const GLenum internalFormat, const GLenum format, unsigned char* data = nullptr);
		GLenum GetTarget() const;
	private:
		TextureDesc m_Description;
		uint32_t m_RendererID = 0;
	};
}

#endif // !TEXTURE_H