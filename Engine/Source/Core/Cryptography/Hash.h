#ifndef HASH_H_
#define HASH_H_

namespace fe {
	/// <summary>
	/// Basic hash function implementing the cyclic redundancy check hashing method.
	/// </summary>
	class Hash
	{
	public:
		static uint32_t GenerateFNVHash(const char* str);
		static uint32_t GenerateFNVHash(const std::string& string);

		static uint32_t CRC32(const char* str);
		static uint32_t CRC32(const std::string& string);
	};
}

#endif