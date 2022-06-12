#ifndef UUID_H
#define UUID_H

#include <xhash>

namespace fe {
	class UUID
	{
	public:
		UUID();
		UUID(uint64_t uuid);
		UUID(const UUID& other);

		operator uint64_t () { return m_UUID; }
		operator const uint64_t() const { return m_UUID; }
	private:
		uint64_t m_UUID;
	};

	class UUID32
	{
	public:
		UUID32();
		UUID32(uint32_t uuid);
		UUID32(const UUID32& other);

		operator uint32_t () { return m_UUID; }
		operator const uint32_t() const { return m_UUID; }
	private:
		uint32_t m_UUID;
	};
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

#endif