#ifndef TEXTURE_ASSET_H
#define TEXTURE_ASSET_H

#include "Renderer/Texture.h"

namespace fe {
	class TextureAsset : public Asset {
	public:
		TextureAsset(const std::string& filepath) 
			: Asset(filepath) 
		{
			TextureDesc desc;
			desc.samples = 1;

			m_Texture = Ref<Texture>::Create(desc, filepath);
		}

		Ref<Texture> GetTexture() const {
			return m_Texture;
		}
	private:
		Ref<Texture> m_Texture;
	};
}

#endif // !TEXTURE_ASSET_H