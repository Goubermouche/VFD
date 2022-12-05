#include "pch.h", 
#include "FrameBuffer.h"

#include "glad/glad.h"

namespace vfd {
	FrameBuffer::FrameBuffer(FrameBufferDescription description)
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
		glViewport(0, 0, static_cast<GLsizei>(m_Description.Width), static_cast<GLsizei>(m_Description.Height));
		glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);
	}

	void FrameBuffer::Unbind()
	{
		if (m_Description.Samples > 1) {
			glBindFramebuffer(GL_READ_FRAMEBUFFER, m_RendererID);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_IntermediaryFrameBuffer->GetRendererID());
			glBlitFramebuffer(0, 0, static_cast<GLint>(m_Description.Width), static_cast<GLint>(m_Description.Height), 
				0, 0, static_cast<GLint>(m_Description.Width), static_cast<GLint>(m_Description.Height), GL_COLOR_BUFFER_BIT, GL_LINEAR);
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

	uint32_t FrameBuffer::GetRendererID() const
	{
		return m_RendererID;
	}

	uint32_t FrameBuffer::GetAttachmentRendererID(const uint32_t index)
	{
		return m_Description.Samples > 1
			? m_IntermediaryFrameBuffer->GetAttachmentRendererID(index)
			: m_Attachments[index]->GetRendererID();
	}

	Ref<Texture> FrameBuffer::GetAttachment(const uint32_t index)
	{
		return m_Description.Samples > 1
			? m_IntermediaryFrameBuffer->GetAttachment(index)
			: m_Attachments[index];
	}

	const FrameBufferDescription& FrameBuffer::GetDescription()
	{
		return m_Description;
	}

	void FrameBuffer::Invalidate()
	{
		m_Attachments.clear();

		glGenFramebuffers(1, &m_RendererID);
		glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);

		uint32_t colorAttachmentIndex = 0;

		for (size_t i = 0; i < m_Description.Attachments.size(); i++)
		{
			TextureDescription desc;
			desc.Samples = m_Description.Samples;
			desc.Width = m_Description.Width;
			desc.Height = m_Description.Height;
			desc.Format = m_Description.Attachments[i];

			m_Attachments.emplace_back(Ref<Texture>::Create(desc));

			if (m_Description.Attachments[i] == TextureFormat::Depth) {
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, static_cast<GLenum>(m_Attachments[i]->GetTarget()), m_Attachments[i]->GetRendererID(), 0);
			}
			else {
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + colorAttachmentIndex, static_cast<GLenum>(m_Attachments[i]->GetTarget()), m_Attachments[i]->GetRendererID(), 0);
				colorAttachmentIndex++;
			}
		}

		if (m_Attachments.size() > 1)
		{
			constexpr GLenum buffers[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
			glDrawBuffers(static_cast<GLsizei>(m_Attachments.size()), buffers);
		}
		else if (m_Attachments.empty())
		{
			glDrawBuffer(GL_NONE);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Create an intermediary FBO for blitting (multi sample only)
		if (m_Description.Samples > 1) {
			FrameBufferDescription desc;
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
			// TODO: multi sampling check
			glReadBuffer(GL_COLOR_ATTACHMENT0 + index);
			uint64_t pixelData;
			glReadPixels(static_cast<GLint>(x), static_cast<GLint>(y), 1, 1, GL_RED_INTEGER, GL_INT, &pixelData);

			return static_cast<uint32_t>(pixelData);
		}
		return 0;

		//return m_IntermediaryFrameBuffer->ReadPixel(index, x, y);
	}
}