#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H

namespace fe {
	enum class FrameBufferTextureFormat {
		None = 0,
		RGBA8,                  // color
		RedInt,	                // color + custom value (used for object picking & sampling the frame buffer)
		Depth24Stencil8,        // depth / stencil
		Depth = Depth24Stencil8 // default
	};

	struct FrameBufferTextureDesc {
		FrameBufferTextureDesc() = default;
		FrameBufferTextureDesc(const FrameBufferTextureFormat format)
			: textureFormat(format)
		{}

		FrameBufferTextureFormat textureFormat = FrameBufferTextureFormat::None;
	};

	struct FrameBufferAttachmentDesc {
		FrameBufferAttachmentDesc() = default;
		FrameBufferAttachmentDesc(const std::initializer_list<FrameBufferTextureDesc> attachments)
			: attachments(attachments)
		{}

		std::vector<FrameBufferTextureDesc> attachments;
	};

	struct FrameBufferDesc {
		uint16_t width = 0;
		uint16_t height = 0;
		FrameBufferAttachmentDesc attachments;
		size_t samples = 1; // TODO: fix antialiasing.

		bool swapChainTarget = false;
	};

	/// <summary>
	/// Simple FBO class, can be used to store data or, more commonly, become the render target.
	/// </summary>
	class FrameBuffer : public RefCounted
	{
	public:
		FrameBuffer(const FrameBufferDesc& description);
		virtual ~FrameBuffer();

		void Invalidate();
		void Resize(uint32_t width, uint32_t height);
		void ClearAttachment(uint32_t attachmentIndex, uint16_t value) const;

		uint32_t GetRendererID() const;
		uint32_t GetColorDescriptionRendererID(uint32_t index) const;

		void Bind() const;
		static void Unbind();

		FrameBufferDesc& GetDescription();
		static int ReadPixel(uint32_t attachmentIndex, uint16_t x, uint16_t y);
	private:
		uint32_t m_RendererID = 0;
		FrameBufferDesc m_Description;
		std::vector<FrameBufferTextureDesc> m_ColorAttachmentDescriptions;
		FrameBufferTextureDesc m_DepthAttachmentDescription = FrameBufferTextureFormat::None;
		std::vector<uint32_t> m_ColorAttachments;
		uint32_t m_DepthAttachment = 0;

		friend class Scene;
	};
}

#endif // !FRAME_BUFFER_H

