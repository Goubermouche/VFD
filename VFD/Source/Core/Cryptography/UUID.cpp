#include "pch.h"
#include "UUID.h"

#include <random>

namespace vfd {
	static std::random_device s_RandomDevice;

	static std::mt19937_64 eng64(s_RandomDevice());
	static std::uniform_int_distribution<uint64_t> s_UniformDistribution;

	static std::mt19937 eng32(s_RandomDevice());
	static std::uniform_int_distribution<uint32_t> s_UniformDistribution32;

	UUID::UUID()
		: value(s_UniformDistribution(eng64))
	{}

	UUID::UUID(const uint64_t uuid)
		: value(uuid)
	{}

	UUID32::UUID32()
		: value(s_UniformDistribution32(eng32))
	{}

	UUID32::UUID32(const uint32_t uuid)
		: value(uuid)
	{}
}
