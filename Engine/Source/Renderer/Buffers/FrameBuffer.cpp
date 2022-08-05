#include "pch.h"
#include "FrameBuffer.h"

#include <Glad/glad.h>

namespace fe {
	static const uint32_t s_MaxFramebufferSize = 8192;

	static GLenum TextureTarget(const bool multiSampled)
	{
		return multiSampled ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
	}

	static void CreateTextures(const bool multiSampled, uint32_t* outID, const uint32_t count)
	{
		glCreateTextures(TextureTarget(multiSampled), count, outID);
	}

	static void BindTexture(const bool multiSampled, uint32_t id)
	{
		glBindTexture(TextureTarget(multiSampled), id);
	}

	static void AttachColorTexture(const uint32_t id, const uint16_t samples, const GLenum internalFormat, const GLenum format, const uint32_t width, const uint32_t height, const uint16_t index)
	{
		const bool multiSampled = samples > 1;

		if (multiSampled) {
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

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + index, TextureTarget(multiSampled), id, 0);
	}

	static void AttachDepthTexture(const uint32_t id, const uint16_t samples, const GLenum format, const GLenum attachmentType, const uint32_t width, const uint32_t height)
	{
		const bool multiSampled = samples > 1;
		if (multiSampled) {
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

		glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentType, TextureTarget(multiSampled), id, 0);
	}

	static bool IsDepthFormat(const FrameBufferTextureFormat format)
	{
		switch (format)
		{
		case FrameBufferTextureFormat::Depth24Stencil8:  return true;
		}

		return false;
	}

	static GLenum FBTextureFormatToGL(const FrameBufferTextureFormat format)
	{
		switch (format)
		{
		case FrameBufferTextureFormat::RGBA8:  return GL_RGBA8;
		case FrameBufferTextureFormat::RedInt: return GL_RED_INTEGER;
		}

		ASSERT(false, "unknown texture format!");
		return 0;
	}

	FrameBuffer::FrameBuffer(const FrameBufferDesc& description)
		: m_Description(description)
	{
		for (auto desc : m_Description.Attachments.Attachments) {
			if (!IsDepthFormat(desc.TextureFormat)) {
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

		const bool multiSample = m_Description.Samples > 1;

		// Attachments
		if (m_ColorAttachmentDescriptions.empty() == false)
		{
			m_ColorAttachments.resize(m_ColorAttachmentDescriptions.size());
			CreateTextures(multiSample, m_ColorAttachments.data(), m_ColorAttachments.size());

			for (size_t i = 0; i < m_ColorAttachments.size(); i++) {
				BindTexture(multiSample, m_ColorAttachments[i]);
				switch (m_ColorAttachmentDescriptions[i].TextureFormat)
				{
				case FrameBufferTextureFormat::RGBA8:
					AttachColorTexture(m_ColorAttachments[i], m_Description.Samples, GL_RGBA8, GL_RGBA, m_Description.Width, m_Description.Height, i);
					break;
				case FrameBufferTextureFormat::RedInt:
					AttachColorTexture(m_ColorAttachments[i], m_Description.Samples, GL_R32I, GL_RED_INTEGER, m_Description.Width, m_Description.Height, i);
					break;
				}
			}
		}

		if (m_DepthAttachmentDescription.TextureFormat != FrameBufferTextureFormat::None)
		{
			CreateTextures(multiSample, &m_DepthAttachment, 1);
			BindTexture(multiSample, m_DepthAttachment);

			switch (m_DepthAttachmentDescription.TextureFormat) {
			case FrameBufferTextureFormat::Depth24Stencil8:
				AttachDepthTexture(m_DepthAttachment, m_Description.Samples, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT, m_Description.Width, m_Description.Height);
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

	void FrameBuffer::Resize(const uint32_t width, const uint32_t height)
	{
		if (width == 0 || height == 0 || width > s_MaxFramebufferSize || height > s_MaxFramebufferSize) {
			return;
		}

		m_Description.Width = width;
		m_Description.Height = height;

		Invalidate();
	}

	void FrameBuffer::ClearAttachment(const uint32_t attachmentIndex, const uint16_t value) const
	{
		auto& desc = m_ColorAttachmentDescriptions[attachmentIndex];
		glClearTexImage(m_ColorAttachments[attachmentIndex], 0, FBTextureFormatToGL(desc.TextureFormat), GL_INT, &value);
	}

	void FrameBuffer::Bind() const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);
		glViewport(0, 0, m_Description.Width, m_Description.Height);
	}

	void FrameBuffer::Unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	FrameBufferDesc& FrameBuffer::GetDescription()
	{
		return m_Description;
	}

	int FrameBuffer::ReadPixel(const uint32_t attachmentIndex, const uint16_t x, const uint16_t y)
	{
		glReadBuffer(GL_COLOR_ATTACHMENT0 + attachmentIndex);
		uint64_t pixelData;
		glReadPixels(x, y, 1, 1, GL_RED_INTEGER, GL_INT, &pixelData);

		return pixelData;
	}
}