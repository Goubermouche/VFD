#include "pch.h"
#include "AssetManager.h"

namespace fe {
	UUID32 AssetManager::GetAssetIDFromFilepath(const std::string& filepath)
	{
		for (const auto& [key, value] : m_AssetRegistry) {
			if (value->GetSourceFilepath() == filepath) {
				return key;
			}
		}

		ASSERT("asset registry does not contain an asset with the specified file path! (" + filepath + ")");
		return 0;
	}
}
