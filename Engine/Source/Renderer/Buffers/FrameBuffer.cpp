#include "pch.h", 
#include "FrameBuffer.h"

#include "glad/glad.h"

namespace fe {
	FrameBuffer::FrameBuffer(FrameBufferDesc description)
		: m_Description(std::move(description))
	{
		Invalidate();
	}

	FrameBuffer::~FrameBuffer()
	{
		m_Attachments.clear();
		glDeleteFramebuffers(1, &m_RendererID);
	}

	void FrameBuffer::Bind() const
	{
		glViewport(0, 0, m_Description.Width, m_Description.Height);
		glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);
	}

	void FrameBuffer::Unbind()
	{
		if (m_Description.Samples > 1) {
			glBindFramebuffer(GL_READ_FRAMEBUFFER, m_RendererID);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_IntermediaryFrameBuffer->GetRendererID());
			glBlitFramebuffer(0, 0, m_Description.Width, m_Description.Height, 0, 0, m_Description.Width, m_Description.Height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void FrameBuffer::Resize(const uint32_t width, const uint32_t height)
	{
		m_Description.Width = width;
		m_Description.Height = height;

		if (m_Description.Samples > 1) {
			m_IntermediaryFrameBuffer->Resize(width, height);
		}

		glDeleteFramebuffers(1, &m_RendererID);

		Invalidate();
	}

	void FrameBuffer::Invalidate()
	{
		m_Attachments.clear();

		glGenFramebuffers(1, &m_RendererID);
		glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);

		uint32_t colorAttachmentIndex = 0;

		for (size_t i = 0; i < m_Description.Attachments.size(); i++)
		{
			TextureDesc desc;
			desc.Samples = m_Description.Samples;
			desc.Width = m_Description.Width;
			desc.Height = m_Description.Height;
			desc.Format = m_Description.Attachments[i];

			m_Attachments.emplace_back(Ref<Texture>::Create(desc));

			if (m_Description.Attachments[i] == TextureFormat::Depth) {
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, (uint32_t)m_Attachments[i]->GetTarget(), m_Attachments[i]->GetRendererID(), 0);
			}
			else {
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + colorAttachmentIndex, (uint32_t)m_Attachments[i]->GetTarget(), m_Attachments[i]->GetRendererID(), 0);
				colorAttachmentIndex++;
			}
		}

		if (m_Attachments.size() > 1)
		{
			GLenum buffers[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
			glDrawBuffers(m_Attachments.size(), buffers);
		}
		else if (m_Attachments.empty())
		{
			glDrawBuffer(GL_NONE);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Create an intermediary FBO for blitting (multi sample only)
		if (m_Description.Samples > 1) {
			FrameBufferDesc desc;
			desc.Samples = 1;
			desc.Width = m_Description.Width;
			desc.Height = m_Description.Height;
			desc.Attachments = { TextureFormat::RGBA8 };

			m_IntermediaryFrameBuffer = Ref<FrameBuffer>::Create(desc);
		}
	}

	uint32_t FrameBuffer::ReadPixel(const uint32_t index, const uint32_t x, const uint32_t y) const
	{
		if (m_Description.Samples == 1) {
			// TODO: multisampling check
			glReadBuffer(GL_COLOR_ATTACHMENT0 + index);
			uint64_t pixelData;
			glReadPixels(x, y, 1, 1, GL_RED_INTEGER, GL_INT, &pixelData);

			return (uint32_t)pixelData;
		}
		return 0;
		//return m_IntermediaryFrameBuffer->ReadPixel(index, x, y);
	}
}