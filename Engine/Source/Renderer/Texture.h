#ifndef TEXTURE_H
#define TEXTURE_H

namespace fe {
	enum class TextureFormat {
		None                 = 0,
		RGBA8                = 0x1908, // color
		RedInt               = 0x8D94, // color + custom value (used for object picking & sampling the frame buffer)
		Depth24Stencil8      = 0x821A, // depth / stencil
		Depth                = 0x1902  // default
	};

	enum class TextureParameterName {
		None                 = 0,
		MinFilter            = 0x2801,
		MagFilter            = 0x2800,
		WrapR                = 0x8072,
		WrapS                = 0x2802,
		WrapT                = 0x2803
	};				      

	enum class TextureTarget
	{
		Texture2D            = 0x0DE1,
		Texture2dMultiSample = 0x9100
	};

	enum class TextureParameterValue {
		None                 = 0,
		// MinFilter, MagFilter
		Nearest              = 0x2600, 
		Linear               = 0x2601,
		// WrapR, WrapS, WrapT
		ClampToEdge          = 0x812F,
		Repeat               = 0x2901
	};

	struct TextureParameter {
		TextureParameterName Name = TextureParameterName::None;
		TextureParameterValue Value = TextureParameterValue::None;
	};

	struct TextureDesc {
		uint32_t Samples = 1;
		uint32_t Width = 0;
		uint32_t Height = 0;
		TextureFormat Format = TextureFormat::RGBA8;

		std::vector<TextureParameter> Parameters = {
			{ TextureParameterName::MinFilter, TextureParameterValue::Linear }
		};
	};

	class Texture : public RefCounted
	{
	public:
		Texture(TextureDesc description);
		Texture(TextureDesc description, const std::string& filepath);
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
		void Attach(const unsigned char* data = nullptr);
	private:
		TextureDesc m_Description;
		uint32_t m_RendererID = 0;
	};
}

#endif // !TEXTURE_H