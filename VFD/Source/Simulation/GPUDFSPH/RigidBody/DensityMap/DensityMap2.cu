#include "pch.h"
#include "DensityMap2.cuh"

#include "Compute/ComputeHelper.h"
#include <thrust/host_vector.h>

namespace vfd {
	template<class T>
	__host__ bool Read(std::streambuf& buf, T& val)
	{
		static_assert(std::is_standard_layout<T>{}, "data is not standard layout");
		auto bytes = sizeof(T);
		return buf.sgetn(reinterpret_cast<char*>(&val), bytes) == bytes;
	}

	DensityMap2::DensityMap2(const std::string& meshSourceFile)
	{
		//auto in = std::ifstream(meshSourceFile, std::ios::binary);

		//if (!in.good())
		//{
		//	std::cerr << "ERROR: grid can not be loaded. Input file does not exist!" << std::endl;
		//	return;
		//}

		//Read(*in.rdbuf(), m_Domain);
		//Read(*in.rdbuf(), m_Resolution);
		//Read(*in.rdbuf(), m_CellSize);
		//Read(*in.rdbuf(), m_CellSizeInverse);
		//size_t cellCount;
		//Read(*in.rdbuf(), cellCount);
		//size_t fieldCount;
		//Read(*in.rdbuf(), fieldCount);

		{
			//auto a = std::size_t{};
			//Read(*in.rdbuf(), a);
			//auto b = std::size_t{};
			//Read(*in.rdbuf(), b);
			//unsigned int size = a * b;

			//m_FieldCount = a;
			//std::vector<double> nodes(size);

			//for (int i = 0; i < b; ++i)
			//{
			//	Read(*in.rdbuf(), nodes[i]);
			//}

			//Read(*in.rdbuf(), b);

			//for (int i = 0; i < b; ++i)
			//{
			//	Read(*in.rdbuf(), nodes[b + i]);
			//}

			//m_Nodes = nodes;
		}

		{
			//auto a = std::size_t{};
			//Read(*in.rdbuf(), a);
			//auto b = std::size_t{};
			//Read(*in.rdbuf(), b);
			//unsigned int size = a * b * 32;

			//m_FieldCount = a;

			////unsigned int index = 0;
			////for (int i = 0; i < b; ++i)
			////{
			////	std::array<unsigned int, 32> cell;
			////	Read(*in.rdbuf(), cell);

			////	for (int j = 0; j < 32; ++j)
			////	{
			////		cells[index++] = cell[j];
			////	}
			////}

			////Read(*in.rdbuf(), b);

			////for (int i = 0; i < b; ++i)
			////{
			////	std::array<unsigned int, 32> cell;
			////	Read(*in.rdbuf(), cell);

			////	for (int j = 0; j < 32; ++j)
			////	{
			////		cells[index++] = cell[j];
			////	}
			////}

			////std::vector<unsigned int> cells(1000);

			////                                              1152000

			//std::vector<unsigned int> v;
			//v.resize(1152000);
			//m_Cells = v;
		}

		//{
		//	auto a = std::size_t{};
		//	Read(*in.rdbuf(), a);
		//	auto b = std::size_t{};
		//	Read(*in.rdbuf(), b);
		//	unsigned int size = a * b;

		//	m_FieldCount = a;
		//	std::vector<unsigned int> cellMap(size);

		//	for (int i = 0; i < b; ++i)
		//	{
		//		Read(*in.rdbuf(), cellMap[i]);
		//	}

		//	Read(*in.rdbuf(), b);

		//	for (int i = 0; i < b; ++i)
		//	{
		//		Read(*in.rdbuf(), cellMap[b + i]);
		//	}

		//	m_CellMap = cellMap;
		//}

		// in.close();

		auto nodes = std::vector<double>();
		auto cells = std::vector<unsigned int>();
		auto cellMap = std::vector<unsigned int>();

		nodes.resize(36000, 1.0);
		cells.resize(1152000, 2);// 1152000
		cellMap.resize(36000, 3); 

		m_Nodes = nodes;
		m_Cells = cells;
		m_CellMap = cellMap;
	}

	DensityMapDeviceData* DensityMap2::GetDeviceData()
	{
		auto* temp = new DensityMapDeviceData();
		DensityMapDeviceData* device;

		temp->m_Domain = m_Domain;
		temp->m_Resolution = m_Resolution;
		temp->m_CellSize = m_CellSize;
		temp->m_CellSizeInverse = m_CellSizeInverse;
		temp->m_FieldCount = m_FieldCount;

		temp->m_NodeCount = static_cast<unsigned int>(m_Nodes.size()) / m_FieldCount;
		temp->m_CellCount = static_cast<unsigned int>(m_Cells.size()) / m_FieldCount / 32u;
		temp->m_CellMapCount = static_cast<unsigned int>(m_CellMap.size()) / m_FieldCount;

		temp->m_Nodes = ComputeHelper::GetPointer(m_Nodes);
		temp->m_Cells = ComputeHelper::GetPointer(m_Cells);
		temp->m_CellMap = ComputeHelper::GetPointer(m_CellMap);

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(DensityMapDeviceData)))
		COMPUTE_SAFE(cudaMemcpy(device, temp, sizeof(DensityMapDeviceData), cudaMemcpyHostToDevice))

		delete temp;
		return device;
	}
}
