#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>

namespace vfd {
	enum class TextureFormat {
		None = 0,
		RGBA8,    // color
		RedInt,   // color + custom value (used for object picking & sampling the frame buffer)
		Depth,    // default
	};

	enum class TextureParameterName {
		None        = 0,
		MinFilter   = GL_TEXTURE_MIN_FILTER,
		MagFilter   = GL_TEXTURE_MAG_FILTER,
		WrapR       = GL_TEXTURE_WRAP_R,
		WrapS       = GL_TEXTURE_WRAP_S,
		WrapT       = GL_TEXTURE_WRAP_T,
		CompareMode = GL_TEXTURE_COMPARE_MODE,
		CompareFunc = GL_TEXTURE_COMPARE_FUNC
	};				      

	enum class TextureTarget
	{
		Texture2D            = GL_TEXTURE_2D,
		Texture2dMultiSample = GL_TEXTURE_2D_MULTISAMPLE
	};

	enum class TextureParameterValue {
		None        = GL_NONE,
		// MinFilter, MagFilter
		Nearest     = GL_NEAREST,
		Linear      = GL_LINEAR,
		// WrapR, WrapS, WrapT
		ClampToEdge = GL_CLAMP_TO_EDGE,
		Repeat      = GL_REPEAT,
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

		uint32_t GetRendererID() const;
		const uint32_t& GetWidth() const;
		const uint32_t& GetHeight() const;
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