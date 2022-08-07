#include "pch.h"
#include "Texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Utility/FileSystem.h"

namespace fe {
	Texture::Texture(TextureDesc description)
		: m_Description(std::move(description))
	{
		Init();
	}

	Texture::Texture(TextureDesc description, const std::string& filepath)
		: m_Description(std::move(description))
	{
		ASSERT(FileExists(filepath), "image file does not exist (" + filepath + ")!");
		
		int imageWidth = 0;
		int imageHeight = 0;

		unsigned char* imageData = stbi_load(filepath.c_str(), &imageWidth, &imageHeight, nullptr, 4);

		if (imageData == nullptr) {
			ASSERT("no image data! (" + filepath + ")!");
		}

		m_Description.Width = imageWidth;
		m_Description.Height = imageHeight;

		Init(imageData);

		stbi_image_free(imageData);

		LOG("texture created successfully (" + filepath + ")", "renderer][shader", ConsoleColor::Green);
	}

	Texture::~Texture()
	{
		glDeleteTextures(1, &m_RendererID);
	}

	void Texture::Bind() const
	{
		glBindTexture((uint32_t)GetTarget(), m_RendererID);
	}

	void Texture::Unbind() const 
	{
		glBindTexture((uint32_t)GetTarget(), 0);
	}

	void Texture::Init(const unsigned char* data)
	{
		switch (m_Description.Format)
		{
		case TextureFormat::Depth:
			Attach(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, data);
			break;
		case TextureFormat::RGBA8:
			Attach(GL_RGBA8, GL_RGBA, data);
			break;
		case TextureFormat::RedInt:
			Attach(GL_R32I, GL_RED_INTEGER, data);
			break;
		}
	}

	void Texture::Attach(GLenum internalFormat, GLenum format,  const unsigned char* data)
	{
		glGenTextures(1, &m_RendererID);
		Bind();

		const bool multiSampled = m_Description.Samples > 1;
		const TextureTarget target = GetTarget();

		if (multiSampled) {
			glTexImage2DMultisample((uint32_t)target, m_Description.Samples, internalFormat, m_Description.Width, m_Description.Height, GL_FALSE);
		}
		else {
			glTexImage2D((uint32_t)target, 0, internalFormat, m_Description.Width, m_Description.Height, 0, format, GL_UNSIGNED_BYTE, data);

			for (const auto& param : m_Description.Parameters)
			{
				glTexParameteri((uint32_t)target, (GLenum)param.Name, (GLint)param.Value);
			}
		}
	}

	TextureTarget Texture::GetTarget() const
	{
		//if (m_Description.Format == TextureFormat::RedInt) {
		//	return TextureTarget::Texture2D;
		//}
		
		return m_Description.Samples > 1 ? TextureTarget::Texture2dMultiSample : TextureTarget::Texture2D;
	}
}