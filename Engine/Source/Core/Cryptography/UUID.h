#ifndef UUID_H
#define UUID_H

#include <xhash>
#include <cereal.hpp>

namespace fe {
	/// <summary>
	/// Representation of a unique identifier (64 bit). 
	/// </summary>
	class UUID
	{
	public:
		UUID();
		UUID(uint64_t uuid); 
		UUID(const UUID& other) = default;
		~UUID() = default;

		[[nodiscard]]
		operator uint64_t () { 
			return value; 
		}

		operator const uint64_t() const {
			return value;
		}
	public:
		uint64_t value = 0;
	};

	// Cereal save function
	template<class Archive>
	void serialize(Archive& archive, UUID& uuid)
	{
		archive(cereal::make_nvp("UUID64", uuid.value));
	}

	/// <summary>
	/// Representation of a unique identifier (32 bit). 
	/// </summary>
	class UUID32
	{
	public:
		UUID32();
		UUID32(uint32_t uuid);
		UUID32(const UUID32& other) = default;
		~UUID32() = default;

		[[nodiscard]]
		operator uint32_t() { 
			return value;
		}

		operator const uint32_t() const {
			return value;
		}
	public:
		uint32_t value = 0;
	};

	// Cereal save function
	template<class Archive>
	void serialize(Archive& archive, UUID32& uuid)
	{
		archive(cereal::make_nvp("UUID32", uuid.value));
	}
}

namespace std {
	template <>
	struct hash<fe::UUID>
	{
		std::size_t operator()(const fe::UUID& uuid) const
		{
			return uuid;
		}
	};

	template <>
	struct hash<fe::UUID32>
	{
		std::size_t operator()(const fe::UUID32& uuid) const
		{
			return hash<uint32_t>()((uint32_t)uuid);
		}
	};
}

#endif // !UUID_H