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
		glViewport(0, 0, m_Description.Width, m_Description.Height);
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

		for (size_t i = 0; i < m_Description.Attachments.size(); i++)
		{
			TextureDesc desc;
			desc.Samples = m_Description.Samples;
			desc.Width = m_Description.Width;
			desc.Height = m_Description.Height;
			desc.Format = m_Description.Attachments[i];

			m_Attachments.emplace_back(Ref<Texture>::Create(desc));

			// TODO: Add support for multiple color attachments
			switch (m_Description.Attachments[i])
			{
			case TextureFormat::RGBA8:
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (uint32_t)m_Attachments[i]->GetTarget(), m_Attachments[i]->GetRendererID(), 0);
				break;
			case TextureFormat::Depth:
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, (uint32_t)m_Attachments[i]->GetTarget(), m_Attachments[i]->GetRendererID(), 0);
				break;
			}
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Create an intermediary FBO for blitting
		if (m_Description.Samples > 1) {
			FrameBufferDesc desc;
			desc.Samples = 1;
			desc.Width = m_Description.Width;
			desc.Height = m_Description.Height;
			desc.Attachments = { TextureFormat::RGBA8 };

			m_IntermediaryFrameBuffer = Ref<FrameBuffer>::Create(desc);
		}
	}
}