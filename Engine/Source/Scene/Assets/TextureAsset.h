#ifndef TEXTURE_ASSET_H
#define TEXTURE_ASSET_H

namespace fe {
	class TextureAsset : public Asset {
	public:
		TextureAsset(const std::string& filepath) 
			: Asset(filepath) 
		{}
	private:

	};
}

#endif // !TEXTURE_ASSET_H