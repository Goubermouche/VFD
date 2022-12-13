#include "pch.h"
#include "RigidBodyData.h"
#include "RigidBodyObject.h"

namespace vfd
{
	template<class T>
	__host__ bool read(std::streambuf& buf, T& val)
	{
		static_assert(std::is_standard_layout<T>{}, "data is not standard layout");
		auto bytes = sizeof(T);
		return buf.sgetn(reinterpret_cast<char*>(&val), bytes) == bytes;
	}

	// TEMP
	__host__ RigidBodyData::RigidBodyData(const RigidBodyDescription& desc)
		: Transform(desc.Transform)
	{
		auto in = std::ifstream("Resources/b.cdm", std::ios::binary);

		if (!in.good())
		{
			std::cerr << "ERROR: grid can not be loaded. Input file does not exist!" << std::endl;
			return;
		}

		read(*in.rdbuf(), Domain);
		read(*in.rdbuf(), Resolution);
		read(*in.rdbuf(), CellSize);
		read(*in.rdbuf(), CellSizeInverse);
		read(*in.rdbuf(), CellCount);
		size_t fieldCount;
		read(*in.rdbuf(), fieldCount);

		std::cout << "Domain min(" << Domain.min.x << ", " << Domain.min.y << ", " << Domain.min.z << ")\n";
		std::cout << "Domain max(" << Domain.max.x << ", " << Domain.max.y << ", " << Domain.max.z << ")\n";
		std::cout << "Resolution (" << Resolution.x << ", " << Resolution.y << ", " << Resolution.z << ")\n";
		std::cout << "CellSize (" << CellSize.x << ", " << CellSize.y << ", " << CellSize.z << ")\n";
		std::cout << "CellSizeInverse (" << CellSizeInverse.x << ", " << CellSizeInverse.y << ", " << CellSizeInverse.z << ")\n";
		std::cout << "CellCount " << CellCount << "\n";
		
		{
			auto a = std::size_t{};
			read(*in.rdbuf(), a);
			auto b = std::size_t{};
			read(*in.rdbuf(), b);
			unsigned int size = a * b;

			NodeCount = a;
			NodeElementCount = b;

			Nodes = new double[size];

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), Nodes[i]);
			}

			read(*in.rdbuf(), b);

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), Nodes[b + i]);
			}
		}

		{
			auto a = std::size_t{};
			read(*in.rdbuf(), a);
			auto b = std::size_t{};
			read(*in.rdbuf(), b);
			unsigned int size = a * b * 32;

			CellCount = a;
			CellElementCount = b;

			Cells = new unsigned int[size];

			// 18 000
			for (int i = 0; i < b; ++i)
			{
				// b
				// ..
				std::array<unsigned int, 32> cell;
				read(*in.rdbuf(), cell);
				memcpy(Cells + i * 32, cell.data(), 32 *sizeof(unsigned int));
			}

			read(*in.rdbuf(), b);

			for (int i = 0; i < b; ++i)
			{
				// b
				// ..
				std::array<unsigned int, 32> cell;
				read(*in.rdbuf(), cell);
				memcpy(Cells + b + i * 32, cell.data(), 32 * sizeof(unsigned int));
			}
		}
	
		{
			auto a = std::size_t{};
			read(*in.rdbuf(), a);
			auto b = std::size_t{};
			read(*in.rdbuf(), b);
			unsigned int size = a * b;

			CellMapCount = a;
			CellMapElementCount = b;

			CellMap = new unsigned int[size];

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), CellMap[i]);
			}

			read(*in.rdbuf(), b);

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), CellMap[b + i]);
			}
		}

		in.close();
	}
}
