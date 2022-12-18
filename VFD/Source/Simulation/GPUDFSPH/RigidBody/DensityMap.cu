#include "pch.h"
#include "DensityMap.cuh"

namespace vfd
{
	template<class T>
	__host__ bool read(std::streambuf& buf, T& val)
	{
		static_assert(std::is_standard_layout<T>{}, "data is not standard layout");
		auto bytes = sizeof(T);
		return buf.sgetn(reinterpret_cast<char*>(&val), bytes) == bytes;
	}

	__host__ DensityMap::DensityMap(const std::string& meshSourceFile)
	{
		// TEMP initialization 
		auto in = std::ifstream(meshSourceFile, std::ios::binary);

		if (!in.good())
		{
			std::cerr << "ERROR: grid can not be loaded. Input file does not exist!" << std::endl;
			return;
		}

		read(*in.rdbuf(), m_Domain);
		read(*in.rdbuf(), m_Resolution);
		read(*in.rdbuf(), m_CellSize);
		read(*in.rdbuf(), m_CellSizeInverse);
		read(*in.rdbuf(), m_CellCount);
		size_t fieldCount;
		read(*in.rdbuf(), fieldCount);

		{
			auto a = std::size_t{};
			read(*in.rdbuf(), a);
			auto b = std::size_t{};
			read(*in.rdbuf(), b);
			unsigned int size = a * b;

			m_NodeCount = a;
			m_NodeElementCount = b;
			m_Nodes = new double[size];

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), m_Nodes[i]);
			}

			read(*in.rdbuf(), b);

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), m_Nodes[b + i]);
			}
		}

		{
			auto a = std::size_t{};
			read(*in.rdbuf(), a);
			auto b = std::size_t{};
			read(*in.rdbuf(), b);
			unsigned int size = a * b * 32;

			m_CellCount = a;
			m_CellElementCount = b;
			m_Cells = new unsigned int[size];

			unsigned int index = 0;
			for (int i = 0; i < b; ++i)
			{
				std::array<unsigned int, 32> cell;
				read(*in.rdbuf(), cell);

				for (int j = 0; j < 32; ++j)
				{
					m_Cells[index++] = cell[j];
				}
			}

			read(*in.rdbuf(), b);

			for (int i = 0; i < b; ++i)
			{
				std::array<unsigned int, 32> cell;
				read(*in.rdbuf(), cell);

				for (int j = 0; j < 32; ++j)
				{
					m_Cells[index++] = cell[j];
				}
			}
		}

		{
			auto a = std::size_t{};
			read(*in.rdbuf(), a);
			auto b = std::size_t{};
			read(*in.rdbuf(), b);
			unsigned int size = a * b;

			m_CellMapCount = a;
			m_CellMapElementCount = b;
			m_CellMap = new unsigned int[size];

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), m_CellMap[i]);
			}

			read(*in.rdbuf(), b);

			for (int i = 0; i < b; ++i)
			{
				read(*in.rdbuf(), m_CellMap[b + i]);
			}
		}

		in.close();
	}

	DensityMap::~DensityMap()
	{
		ERR("free density map")
	}
}