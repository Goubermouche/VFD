#ifndef OPENGL_FRAME_BUFFER_H_
#define OPENGL_FRAME_BUFFER_H_

#include "FluidEngine/Renderer/Buffers/FrameBuffer.h"

namespace fe::opengl {
	class OpenGLFrameBuffer : public FrameBuffer
	{
	public:
		OpenGLFrameBuffer(const FrameBufferSpecification& specification);
		virtual ~OpenGLFrameBuffer();

		virtual void Invalidate() override;
		virtual void Resize(uint32_t width, uint32_t height) override;
		virtual void ClearAttachment(uint32_t attachmentIndex, int value) override;

		virtual uint32_t GetRendererID(uint32_t index = 0) override;

		virtual void Bind() const override;
		virtual void Unind() const override;

		virtual FrameBufferSpecification& GetSpecification() override;
		virtual int ReadPixel(uint32_t attachmentIndex, int x, int y) override;
	private:
		uint32_t m_RendererId = 0;
		FrameBufferSpecification m_Specification;
		std::vector<FrameBufferTextureSpecification> m_ColorAttachmentSpecifications;
		FrameBufferTextureSpecification m_DepthAttachmentSpecification
			= FrameBufferTextureFormat::None;

		std::vector<uint32_t> m_ColorAttachments;
		uint32_t m_DepthAttachment = 0;

		friend class Scene;
	};
}

#endif // !OPENGL_FRAME_BUFFER_H_



