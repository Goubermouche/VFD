#ifndef FRAME_BUFFER_2_H
#define FRAME_BUFFER_2_H

#include "Renderer/Texture.h"

namespace fe {
	struct FrameBufferDesc {
		uint16_t Samples = 1;
		uint32_t Width;
		uint32_t Height;
		std::vector<TextureFormat> Attachments;
	};

	class FrameBuffer : public RefCounted
	{
	public:
		FrameBuffer(FrameBufferDesc description);
		~FrameBuffer();

		void Bind() const;
		void Unbind();

		void Resize(uint32_t width, uint32_t height);

		uint32_t GetRendererID() const {
			return m_RendererID;
		}

		uint32_t GetAttachmentRendererID(const uint32_t index) {
			return m_Description.Samples > 1
				? m_IntermediaryFrameBuffer->GetAttachmentRendererID(index)
				: m_Attachments[index]->GetRendererID();
		}

		Ref<Texture> GetAttachment(const uint32_t index) {
			return m_Description.Samples > 1
				? m_IntermediaryFrameBuffer->GetAttachment(index)
				: m_Attachments[index];
		}

		const FrameBufferDesc& GetDescription() {
			return m_Description;
		}

		uint32_t ReadPixel(uint32_t index, uint32_t x, uint32_t y) const;
	private:
		void Invalidate();
	private:
		uint32_t m_RendererID = 0;
		std::vector<Ref<Texture>> m_Attachments;
		FrameBufferDesc m_Description;

		Ref<FrameBuffer> m_IntermediaryFrameBuffer;
	};
}

#endif // !FRAME_BUFFER_2_H