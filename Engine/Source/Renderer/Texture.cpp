#include "pch.h"
#include "Texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Utility/FileSystem.h"

namespace fe {
	Texture::Texture(const TextureDesc& description)
		: m_Description(description)
	{
		glGenTextures(1, &m_RendererID);
		Bind();

		switch (m_Description.Format)
		{
		case TextureFormat::RGBA8:
			Attach(GL_RGBA8, GL_RGBA);
			break;
		case TextureFormat::RedInt:
			Attach(GL_R32I, GL_RED_INTEGER);
			break;
		case TextureFormat::Depth:
			Attach(GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT);
		}

		LOG("texture created successfully", "renderer][shader", ConsoleColor::Green);
	}

	Texture::Texture(const TextureDesc& description, const std::string& filepath)
		: m_Description(description)
	{
		ASSERT(FileExists(filepath), "file does not exist (" + filepath + ")!");
		
		int imageWidth = 0;
		int imageHeight = 0;

		// TODO: get component count from file type
		unsigned char* imageData = stbi_load(filepath.c_str(), &imageWidth, &imageHeight, NULL, 4);

		if (imageData == nullptr) {
			ASSERT("no image data! (" + filepath + ")!");
		}

		m_Description.Width = imageWidth;
		m_Description.Height = imageHeight;

		glGenTextures(1, &m_RendererID);
		Bind();

		switch (m_Description.Format)
		{
		case TextureFormat::RGBA8:
			Attach(GL_RGBA8, GL_RGBA, imageData);
			break;
		case TextureFormat::RedInt:
			Attach(GL_R32I, GL_RED_INTEGER, imageData);
			break;
		case TextureFormat::Depth:
			Attach(GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT, imageData);
		}

		stbi_image_free(imageData);

		LOG("texture created successfully (" + filepath + ")", "renderer][shader", ConsoleColor::Green);
	}

	void Texture::Bind() const
	{
		glBindTexture(GetTarget(), m_RendererID);
	}

	void Texture::Unbind()
	{
		ASSERT("not implemented!");
	}

	void Texture::Attach(const GLenum internalFormat, const GLenum format, unsigned char* data)
	{
		const bool multisampled = m_Description.samples > 1;
		if (multisampled) {
			glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, m_Description.samples, internalFormat, m_Description.Width, m_Description.Height, GL_FALSE);
		}
		else {
			glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, m_Description.Width, m_Description.Height, 0, format, GL_UNSIGNED_BYTE, data);

			GLenum target = GetTarget();

			for (size_t i = 0; i < m_Description.Parameters.size(); i++)
			{
				glTexParameteri(target, (GLenum)m_Description.Parameters[i].Name, (GLint)m_Description.Parameters[i].Value);
			}
		}
	}

	GLenum Texture::GetTarget() const
	{
		if (m_Description.samples > 1) {
			return GL_TEXTURE_2D_MULTISAMPLE;
		}
		return GL_TEXTURE_2D;
	}
}