#include "pch.h"
#include "Texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Utility/FileSystem.h"
#include <glad/glad.h>

namespace fe {
	Texture::Texture(const std::string& filepath)
	{
		ASSERT(FileExists(filepath), "file does not exist (" + filepath + ")!");

		int imageWidth = 0;
		int imageHeight = 0;

		// TODO: get component count from file type
		unsigned char* imageData = stbi_load(filepath.c_str(), &imageWidth, &imageHeight, NULL, 4);

		if (imageData == nullptr) {
			ASSERT("no image data! (" + filepath + ")!");
		}

		// create a texture identifier
		glGenTextures(1, &m_RendererID);
		glBindTexture(GL_TEXTURE_2D, m_RendererID);

		// setup filtering parameters for display
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		// upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData);
		stbi_image_free(imageData);

		m_Size = { imageWidth , imageHeight };
		LOG("texture created successfully (" + filepath + ")", "renderer][shader", ConsoleColor::Green);

		ERR(m_Size.x);
		ERR(m_Size.y);
	}

	void Texture::Bind() const
	{
		ASSERT("not implemented!");
	}

	void Texture::Unbind()
	{
		ASSERT("not implemented!");
	}
}