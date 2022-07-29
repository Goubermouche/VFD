#include "pch.h"
#include "FrameBuffer.h"

#include <Glad/glad.h>

namespace fe {
	static const uint32_t s_MaxFramebufferSize = 8192;

	static GLenum TextureTarget(bool multisampled)
	{
		return multisampled ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
	}

	static void CreateTextures(bool multisampled, uint32_t* outID, uint32_t count)
	{
		glCreateTextures(TextureTarget(multisampled), count, outID);
	}

	static void BindTexture(bool multisampled, uint32_t id)
	{
		glBindTexture(TextureTarget(multisampled), id);
	}

	static void AttachColorTexture(uint32_t id, uint16_t samples, GLenum internalFormat, GLenum format, uint32_t width, uint32_t height, uint16_t index)
	{
		bool multisampled = samples > 1;

		if (multisampled) {
			glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, internalFormat, width, height, GL_FALSE);
		}
		else {
			glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, nullptr);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + index, TextureTarget(multisampled), id, 0);
	}

	static void AttachDepthTexture(uint32_t id, uint16_t samples, GLenum format, GLenum attachmentType, uint32_t width, uint32_t height)
	{
		bool multisampled = samples > 1;
		if (multisampled) {
			glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, format, width, height, GL_FALSE);
		}
		else {
			glTexStorage2D(GL_TEXTURE_2D, 1, format, width, height);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}

		glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentType, TextureTarget(multisampled), id, 0);
	}

	static bool IsDepthFormat(FrameBufferTextureFormat format)
	{
		switch (format)
		{
		case FrameBufferTextureFormat::Depth24Stencil8:  return true;
		}

		return false;
	}

	static GLenum FBTextureFormatToGL(FrameBufferTextureFormat format)
	{
		switch (format)
		{
		case FrameBufferTextureFormat::RGBA8:       return GL_RGBA8;
		case FrameBufferTextureFormat::RedInt: return GL_RED_INTEGER;
		}

		ASSERT(false, "unknown texture format!")
			return 0;
	}

	FrameBuffer::FrameBuffer(const FrameBufferDesc& description)
		: m_Description(description), m_RendererID(0)
	{
		for (auto desc : m_Description.attachments.attachments) {
			if (!IsDepthFormat(desc.textureFormat)) {
				m_ColorAttachmentDescriptions.emplace_back(desc);
			}
			else {
				m_DepthAttachmentDescription = desc;
			}
		}

		Invalidate();
	}

	FrameBuffer::~FrameBuffer()
	{
		glDeleteFramebuffers(1, &m_RendererID);
		glDeleteTextures(m_ColorAttachments.size(), m_ColorAttachments.data());
		glDeleteTextures(1, &m_DepthAttachment);
	}

	void FrameBuffer::Invalidate()
	{
		if (m_RendererID) {
			glDeleteFramebuffers(1, &m_RendererID);
			glDeleteTextures(m_ColorAttachments.size(), m_ColorAttachments.data());
			glDeleteTextures(1, &m_DepthAttachment);

			m_ColorAttachments.clear();
			m_DepthAttachment = 0;
		}

		glCreateFramebuffers(1, &m_RendererID);
		glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);

		bool multisample = m_Description.samples > 1;

		// Attachments
		if (m_ColorAttachmentDescriptions.size())
		{
			m_ColorAttachments.resize(m_ColorAttachmentDescriptions.size());
			CreateTextures(multisample, m_ColorAttachments.data(), m_ColorAttachments.size());

			for (size_t i = 0; i < m_ColorAttachments.size(); i++) {
				BindTexture(multisample, m_ColorAttachments[i]);
				switch (m_ColorAttachmentDescriptions[i].textureFormat)
				{
				case FrameBufferTextureFormat::RGBA8:
					AttachColorTexture(m_ColorAttachments[i], m_Description.samples, GL_RGBA8, GL_RGBA, m_Description.width, m_Description.height, i);
					break;
				case FrameBufferTextureFormat::RedInt:
					AttachColorTexture(m_ColorAttachments[i], m_Description.samples, GL_R32I, GL_RED_INTEGER, m_Description.width, m_Description.height, i);
					break;
				}
			}
		}

		if (m_DepthAttachmentDescription.textureFormat != FrameBufferTextureFormat::None)
		{
			CreateTextures(multisample, &m_DepthAttachment, 1);
			BindTexture(multisample, m_DepthAttachment);

			switch (m_DepthAttachmentDescription.textureFormat) {
			case FrameBufferTextureFormat::Depth24Stencil8:
				AttachDepthTexture(m_DepthAttachment, m_Description.samples, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT, m_Description.width, m_Description.height);
				break;
			}
		}

		if (m_ColorAttachments.size() > 1) {
			GLenum buffers[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
			glDrawBuffers(m_ColorAttachments.size(), buffers);
		}
		else if (m_ColorAttachments.empty()) {
			// Only depth-pass
			glDrawBuffer(GL_NONE);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void FrameBuffer::Resize(uint32_t width, uint32_t height)
	{
		if (width == 0 || height == 0 || width > s_MaxFramebufferSize || height > s_MaxFramebufferSize) {
			return;
		}

		m_Description.width = width;
		m_Description.height = height;

		Invalidate();
	}

	void FrameBuffer::ClearAttachment(uint32_t attachmentIndex, uint16_t value)
	{
		auto& desc = m_ColorAttachmentDescriptions[attachmentIndex];
		glClearTexImage(m_ColorAttachments[attachmentIndex], 0, FBTextureFormatToGL(desc.textureFormat), GL_INT, &value);
	}

	uint32_t FrameBuffer::GetRendererID()
	{
		return m_RendererID;
	}

	uint32_t FrameBuffer::GetColorDescriptionRendererID(uint32_t index)
	{
		return m_ColorAttachments[index];
	}

	void FrameBuffer::Bind() const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);
		glViewport(0, 0, m_Description.width, m_Description.height);
	}

	void FrameBuffer::Unbind() const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	FrameBufferDesc& FrameBuffer::GetDescription()
	{
		return m_Description;
	}

	int FrameBuffer::ReadPixel(uint32_t attachmentIndex, uint16_t x, uint16_t y)
	{
		glReadBuffer(GL_COLOR_ATTACHMENT0 + attachmentIndex);
		uint64_t pixelData;
		glReadPixels(x, y, 1, 1, GL_RED_INTEGER, GL_INT, &pixelData);

		return pixelData;
	}
}