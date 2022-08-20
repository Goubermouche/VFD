#ifndef TEXTURE_ASSET_H
#define TEXTURE_ASSET_H

#include "Renderer/Texture.h"

namespace fe {
	class TextureAsset : public Asset {
	public:
		TextureAsset(const std::string& filepath, const TextureDescription& description = TextureDescription())
			: Asset(filepath) 
		{
			m_Texture = Ref<Texture>::Create(description, filepath);
		}

		Ref<Texture> GetTexture() const {
			return m_Texture;
		}
	private:
		Ref<Texture> m_Texture;
	};
}

#endif // !TEXTURE_ASSET_H