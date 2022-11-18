#ifndef HASH_H
#define HASH_H

namespace vfd {
	/// <summary>
	/// Basic hash class implementing the cyclic redundancy check hashing method.
	/// </summary>
	class Hash
	{
	public:
		/// <summary>
		/// Hashes a given string. 
		/// </summary>
		/// <param name="str">String to hash.</param>
		/// <returns>Hashed representation of the given string.</returns>
		static uint32_t GenerateFNVHash(const char* str);

		/// <summary>
		/// Hashes a given string. 
		/// </summary>
		/// <param name="str">String to hash.</param>
		/// <returns>Hashed representation of the given string.</returns>
		static uint32_t GenerateFNVHash(const std::string& str);

		static uint32_t CRC32(const char* str);
		static uint32_t CRC32(const std::string& str);
	};
}

#endif // !HASH_H