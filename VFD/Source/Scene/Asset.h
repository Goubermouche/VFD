#ifndef ASSET_H
#define ASSET_H

#include "Core/Cryptography/UUID.h"

namespace vfd {
	class Asset : public RefCounted
	{
	public:
		Asset(const std::string& filepath)
			: m_Filepath(filepath)
		{}

		const std::string& GetSourceFilepath() const {
			return m_Filepath;
		}
	protected:
		std::string m_Filepath;
	};
}

#endif // !ASSET_H