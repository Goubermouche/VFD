#ifndef OPENGL_FRAME_BUFFER_H_
#define OPENGL_FRAME_BUFFER_H_

#include "FluidEngine/Renderer/Buffers/FrameBuffer.h"

namespace fe::opengl {
	class OpenGLFrameBuffer : public FrameBuffer
	{
	public:
		OpenGLFrameBuffer(const FrameBufferDesc& specification);
		virtual ~OpenGLFrameBuffer();

		virtual void Invalidate() override;
		virtual void Resize(uint32_t width, uint32_t height) override;
		virtual void ClearAttachment(uint32_t attachmentIndex, int value) override;

		virtual Ref<FrameBuffer> GetIntermediateFrameBuffer() override;
		virtual uint32_t GetRendererID() override;
		virtual uint32_t GetColorSpecificationRendererID(uint32_t index) override;

		virtual void Bind() const override;
		virtual void Unbind() override;

		virtual FrameBufferDesc& GetSpecification() override;
		virtual int ReadPixel(uint32_t attachmentIndex, int x, int y) override;
	private:
		uint32_t m_RendererID = 0;
		FrameBufferDesc m_Specification;
		std::vector<FrameBufferTextureDesc> m_ColorAttachmentSpecifications;
		FrameBufferTextureDesc m_DepthAttachmentSpecification
			= FrameBufferTextureFormat::None;

		std::vector<uint32_t> m_ColorAttachments;
		uint32_t m_DepthAttachment = 0;

		Ref<OpenGLFrameBuffer> m_IntermediateFrameBuffer = nullptr;

		friend class Scene;


		// Inherited via FrameBuffer
		

		// Inherited via FrameBuffer
	};
}

#endif // !OPENGL_FRAME_BUFFER_H_