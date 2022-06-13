#ifndef FRAME_BUFFER_H_
#define FRAME_BUFFER_H_

namespace fe {
	enum class FrameBufferTextureFormat {
		None = 0,
		RGBA8,                  // color
		RedInt,					// color + custom value (used for object picking & sampling the frame buffer)
		Depth24Stencil8,        // depth / stencil
		Depth = Depth24Stencil8 // default
	};

	struct FrameBufferTextureSpecification {
		FrameBufferTextureSpecification() = default;
		FrameBufferTextureSpecification(FrameBufferTextureFormat format)
			: textureFormat(format) {}

		FrameBufferTextureFormat textureFormat = FrameBufferTextureFormat::None;
	};

	struct FrameBufferAttachmentSpecification {
		FrameBufferAttachmentSpecification() = default;
		FrameBufferAttachmentSpecification(std::initializer_list<FrameBufferTextureSpecification> attachments)
			: attachments(attachments) {}

		std::vector<FrameBufferTextureSpecification> attachments;
	};

	struct FrameBufferSpecification {
		unsigned int width = 0;
		unsigned int height = 0;
		FrameBufferAttachmentSpecification attachments;
		size_t samples = 1; // TODO: fix antialiasing.

		bool swapChainTarget = false;
	};

	class FrameBuffer : public RefCounted
	{
	public:
		virtual ~FrameBuffer() {}

		virtual void Invalidate() = 0;
		virtual void Resize(uint32_t width, uint32_t height) = 0;
		virtual void ClearAttachment(uint32_t attachmentIndex, int value) = 0;

		virtual uint32_t GetRendererID(uint32_t index = 0) = 0;
		virtual FrameBufferSpecification& GetSpecification() = 0;
		virtual int ReadPixel(uint32_t attachmentIndex, int x, int y) = 0;

		virtual void Bind() const = 0;
		virtual void Unind() const = 0;

		static Ref<FrameBuffer> Create(const FrameBufferSpecification& specification);
	};
}

#endif // !FRAME_BUFFER_H_

