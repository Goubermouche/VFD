#include "pch.h"
#include "DensityMap.cuh"

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

	// TEMP
	DensityMap::DensityMap(const std::string& meshSourceFile)
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

		//{
		//	auto a = std::size_t{};
		//	Read(*in.rdbuf(), a);
		//	auto b = std::size_t{};
		//	Read(*in.rdbuf(), b);
		//	unsigned int size = a * b;
		//	// a = 2
		//	std::vector<double> nodes(size);
		//	// b = size
		//	for (unsigned int i = 0; i < b; ++i)
		//	{
		//		Read(*in.rdbuf(), nodes[i]);
		//	}

		//	Read(*in.rdbuf(), b);

		//	for (unsigned int i = 0; i < b; ++i)
		//	{
		//		double val;
		//		Read(*in.rdbuf(), val);
		//		nodes[b + i] = val;
		//	}

		//	m_Nodes = nodes;
		//}

		//{
		//	auto a = std::size_t{};
		//	Read(*in.rdbuf(), a);
		//	auto b = std::size_t{};
		//	Read(*in.rdbuf(), b);
		//	const unsigned int size = a * b * 32ull;
		//	std::vector<unsigned int> cells(size);

		//	unsigned int index = 0;
		//	for (unsigned int i = 0; i < static_cast<unsigned int>(b); ++i)
		//	{
		//		std::array<unsigned int, 32> cell;
		//		Read(*in.rdbuf(), cell);

		//		for (int j = 0; j < 32; ++j)
		//		{
		//			cells[index] = cell[j];
		//			index++;
		//		}
		//	}

		//	Read(*in.rdbuf(), b);

		//	for (unsigned int i = 0; i < static_cast<unsigned int>(b); ++i)
		//	{
		//		std::array<unsigned int, 32> cell;
		//		Read(*in.rdbuf(), cell);

		//		for (int j = 0; j < 32; ++j)
		//		{
		//			cells[index] = cell[j];
		//			index++;
		//		}
		//	}

		//	m_Cells = cells;
		//}

		//{
		//	auto a = std::size_t{};
		//	Read(*in.rdbuf(), a);
		//	auto b = std::size_t{};
		//	Read(*in.rdbuf(), b);
		//	const unsigned int size = a * b;

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

		// m_NodeCount = static_cast<unsigned int>(m_Nodes.size()) / m_FieldCount;
		// m_CellCount = static_cast<unsigned int>(m_Cells.size()) / m_FieldCount / 32u;
		// m_CellMapCount = static_cast<unsigned int>(m_CellMap.size()) / m_FieldCount;
	}

	DensityMap::DensityMap(const BoundingBox<glm::dvec3>& domain, glm::uvec3 resolution)
		: m_Resolution(resolution), m_Domain(domain)
	{
		m_CellSize = m_Domain.Diagonal() / static_cast<glm::dvec3>(m_Resolution);
		m_CellSizeInverse = 1.0 / m_CellSize;
		m_CellCount = glm::compMul(m_Resolution);
	}

	void DensityMap::AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate)
	{
		auto n = m_Resolution;

		auto nv = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);
		auto ne_x = (n[0] + 0) * (n[1] + 1) * (n[2] + 1);
		auto ne_y = (n[0] + 1) * (n[1] + 0) * (n[2] + 1);
		auto ne_z = (n[0] + 1) * (n[1] + 1) * (n[2] + 0);
		auto ne = ne_x + ne_y + ne_z;

		auto n_nodes = nv + 2 * ne;

		m_Nodes.push_back({});
		auto& coeffs = m_Nodes.back();
		coeffs.resize(n_nodes);

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) nowait
			for (int l = 0; l < static_cast<int>(n_nodes); ++l) {
				glm::dvec3 x = IndexToNodePosition(l);
				double& c = coeffs[l];

				if (!predicate || predicate(x)) {
					c = function(x);
				}
				else {
					c = std::numeric_limits<double>::max();
				}
			}
		}

		m_Cells.push_back({});
		auto& cells = m_Cells.back();
		cells.resize(m_CellCount);

		for (unsigned int l = 0; l < m_CellCount; ++l)
		{
			auto k = l / (n[1] * n[0]);
			auto temp = l % (n[1] * n[0]);
			auto j = temp / n[0];
			auto i = temp % n[0];

			auto nx = n[0];
			auto ny = n[1];
			auto nz = n[2];

			auto& cell = cells[l];
			cell[0] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i;
			cell[1] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i + 1;
			cell[2] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i;
			cell[3] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i + 1;
			cell[4] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i;
			cell[5] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i + 1;
			cell[6] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i;
			cell[7] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i + 1;

			auto offset = nv;
			cell[8] = offset + 2 * (nx * (ny + 1) * k + nx * j + i);
			cell[9] = cell[8] + 1;
			cell[10] = offset + 2 * (nx * (ny + 1) * (k + 1) + nx * j + i);
			cell[11] = cell[10] + 1;
			cell[12] = offset + 2 * (nx * (ny + 1) * k + nx * (j + 1) + i);
			cell[13] = cell[12] + 1;
			cell[14] = offset + 2 * (nx * (ny + 1) * (k + 1) + nx * (j + 1) + i);
			cell[15] = cell[14] + 1;

			offset += 2 * ne_x;
			cell[16] = offset + 2 * (ny * (nz + 1) * i + ny * k + j);
			cell[17] = cell[16] + 1;
			cell[18] = offset + 2 * (ny * (nz + 1) * (i + 1) + ny * k + j);
			cell[19] = cell[18] + 1;
			cell[20] = offset + 2 * (ny * (nz + 1) * i + ny * (k + 1) + j);
			cell[21] = cell[20] + 1;
			cell[22] = offset + 2 * (ny * (nz + 1) * (i + 1) + ny * (k + 1) + j);
			cell[23] = cell[22] + 1;

			offset += 2 * ne_y;
			cell[24] = offset + 2 * (nz * (nx + 1) * j + nz * i + k);
			cell[25] = cell[24] + 1;
			cell[26] = offset + 2 * (nz * (nx + 1) * (j + 1) + nz * i + k);
			cell[27] = cell[26] + 1;
			cell[28] = offset + 2 * (nz * (nx + 1) * j + nz * (i + 1) + k);
			cell[29] = cell[28] + 1;
			cell[30] = offset + 2 * (nz * (nx + 1) * (j + 1) + nz * (i + 1) + k);
			cell[31] = cell[30] + 1;
		}

		m_CellMap.push_back({});
		auto& cell_map = m_CellMap.back();
		cell_map.resize(m_CellCount);
		std::iota(cell_map.begin(), cell_map.end(), 0u);

		m_FieldCount++;
	}

	double DensityMap::Interpolate(unsigned int fieldID, const glm::dvec3& point, glm::dvec3* gradient)
	{
		if (m_Domain.Contains(point) == false) {
			return std::numeric_limits<double>::max();
		}

		glm::ivec3 multiIndex = (point - m_Domain.min) * m_CellSizeInverse;
		if (multiIndex.x >= m_Resolution.x) {
			multiIndex.x = m_Resolution.x - 1;
		}
		if (multiIndex.y >= m_Resolution.y) {
			multiIndex.y = m_Resolution.y - 1;
		}
		if (multiIndex.z >= m_Resolution.z) {
			multiIndex.z = m_Resolution.z - 1;
		}

		unsigned int index = MultiToSingleIndex(multiIndex);
		unsigned int index_ = m_CellMap[fieldID][index];
		if (index_ == std::numeric_limits<unsigned int>::max()) {
			return std::numeric_limits<double>::max();
		}

		BoundingBox<glm::dvec3> subDomain = CalculateSubDomain(index);
		index = index_;
		glm::dvec3 d = subDomain.Diagonal();
		glm::dvec3 denom = (subDomain.max - subDomain.min);
		glm::dvec3 c0 = 2.0 / denom;
		glm::dvec3 c1 = (subDomain.max + subDomain.min) / (denom);
		glm::dvec3 xi = (c0 * point - c1);

		auto const& cell = m_Cells[fieldID][index];
		if (!gradient)
		{
			double phi = 0.0;
			auto N = ShapeFunction(xi);
			for (unsigned int j = 0; j < 32; ++j)
			{
				unsigned int v = cell[j];
				double c = m_Nodes[fieldID][v];
				if (c == std::numeric_limits<double>::max())
				{
					return std::numeric_limits<double>::max();
				}

				phi += c * N[j];
			}

			return phi;
		}

		std::array<std::array<double, 3>, 32> dN{};
		auto N = ShapeFunction(xi, &dN);

		double phi = 0.0;
		*gradient = { 0.0, 0.0, 0.0 };

		for (unsigned int j = 0; j < 32; ++j)
		{
			unsigned int v = cell[j];
			double c = m_Nodes[fieldID][v];

			if (c == std::numeric_limits<double>::max())
			{
				*gradient = { 0.0, 0.0, 0.0 };
				return std::numeric_limits<double>::max();
			}

			phi += c * N[j];

			(*gradient).x += c * dN[j][0];
			(*gradient).y += c * dN[j][1];
			(*gradient).z += c * dN[j][2];
		}

		return phi;
	}

	DensityMapDeviceData* DensityMap::GetDeviceData()
	{
		// Flatten nodes
		std::vector<double> nodes;
		const unsigned int nodeCount = m_Nodes[0].size();

		for (const auto& n : m_Nodes)
		{
			if(nodeCount == n.size())
			{
				nodes.insert(nodes.end(), n.begin(), n.end());
			}
			else
			{
				ASSERT("element counts must be equal!")
			}
		}

		// Flatten cells
		std::vector<unsigned int> cells;
		const unsigned int cellCount = m_Cells[0].size();

		for (const auto& c : m_Cells)
		{
			for (const auto& a : c)
			{
				for (int i = 0; i < 32; ++i)
				{
					cells.push_back(a[i]);
				}
			}
		}

		// Flatten the cell map
		std::vector<unsigned int> cellMap;
		const unsigned int cellMapCount = m_CellMap[0].size();

		for (const auto& m : m_CellMap)
		{
			if (cellMapCount == m.size())
			{
				cellMap.insert(cellMap.end(), m.begin(), m.end());
			}
			else
			{
				ASSERT("element counts must be equal!")
			}
		}

		d_Nodes = nodes;
		d_Cells = cells;
		d_CellMap = cellMap;

		auto* temp = new DensityMapDeviceData();
		DensityMapDeviceData* device;

		temp->m_Domain = m_Domain;
		temp->m_Resolution = m_Resolution;
		temp->m_CellSize = m_CellSize;
		temp->m_CellSizeInverse = m_CellSizeInverse;

		temp->m_FieldCount = m_FieldCount;
		temp->m_NodeCount = static_cast<unsigned int>(m_Nodes[0].size());
		temp->m_CellCount = m_CellCount;
		temp->m_CellMapCount = static_cast<unsigned int>(m_CellMap[0].size());

		temp->m_Nodes = ComputeHelper::GetPointer(d_Nodes);
		temp->m_Cells = ComputeHelper::GetPointer(d_Cells);
		temp->m_CellMap = ComputeHelper::GetPointer(d_CellMap);

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(DensityMapDeviceData)))
		COMPUTE_SAFE(cudaMemcpy(device, temp, sizeof(DensityMapDeviceData), cudaMemcpyHostToDevice))

		delete temp;
		return device;
	}

	const BoundingBox<glm::dvec3>& DensityMap::GetBounds() const
	{
		return m_Domain;
	}

	glm::dvec3 DensityMap::IndexToNodePosition(unsigned int i) const
	{
		glm::dvec3 result;
		glm::dvec3 index;

		unsigned int nv = (m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 1);

		glm::ivec3 ne = {
			 (m_Resolution.x + 0) * (m_Resolution.y + 1) * (m_Resolution.z + 1),
			 (m_Resolution.x + 1) * (m_Resolution.y + 0) * (m_Resolution.z + 1),
			 (m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 0)
		};

		if (i < nv)
		{
			index.z = i / (unsigned int)((m_Resolution.y + 1) * (m_Resolution.x + 1));
			unsigned int temp = i % (unsigned int)((m_Resolution.y + 1) * (m_Resolution.x + 1));
			index.y = temp / (m_Resolution.x + 1);
			index.x = temp % (unsigned int)(m_Resolution.x + 1);

			result = (glm::dvec3)m_Domain.min + (glm::dvec3)m_CellSize * index;
		}
		else if (i < nv + 2 * ne.x)
		{
			i -= nv;
			unsigned int e_ind = i / 2;
			index.z = e_ind / ((m_Resolution.y + 1) * m_Resolution.x);
			unsigned int temp = e_ind % (unsigned int)((m_Resolution.y + 1) * m_Resolution.x);
			index.y = temp / m_Resolution.x;
			index.x = temp % (unsigned int)m_Resolution.x;

			result = (glm::dvec3)m_Domain.min + (glm::dvec3)m_CellSize * index;
			result.x += (1.0 + i % 2) / 3.0 * m_CellSize.x;
		}
		else if (i < nv + 2 * (ne.x + ne.y))
		{
			i -= (nv + 2 * ne.x);
			unsigned int e_ind = i / 2;
			index.x = e_ind / ((m_Resolution.z + 1) * m_Resolution.y);
			unsigned int temp = e_ind % (unsigned int)((m_Resolution.z + 1) * m_Resolution.y);
			index.z = temp / m_Resolution.y;
			index.y = temp % (unsigned int)m_Resolution.y;

			result = (glm::dvec3)m_Domain.min + (glm::dvec3)m_CellSize * index;
			result.y += (1.0 + i % 2) / 3.0 * m_CellSize.y;
		}
		else
		{
			i -= (nv + 2 * (ne.x + ne.y));
			unsigned int e_ind = i / 2;
			index.y = e_ind / ((m_Resolution.x + 1) * m_Resolution.z);
			unsigned int temp = e_ind % (unsigned int)((m_Resolution.x + 1) * m_Resolution.z);
			index.x = temp / m_Resolution.z;
			index.z = temp % (unsigned int)m_Resolution.z;

			result = (glm::dvec3)m_Domain.min + (glm::dvec3)m_CellSize * index;
			result.z += (1.0 + i % 2) / 3.0 * m_CellSize.z;
		}

		return result;
	}

	unsigned int DensityMap::MultiToSingleIndex(const glm::uvec3& index) const
	{
		return m_Resolution.y * m_Resolution.x * index.z + m_Resolution.x * index.y + index.x;
	}

	glm::uvec3 DensityMap::SingleToMultiIndex(const unsigned int index) const
	{
		const unsigned int n01 = m_Resolution.x * m_Resolution.y;
		unsigned int k = index / n01;
		const unsigned int temp = index % n01;
		double j = temp / m_Resolution.x;
		double i = temp % m_Resolution.x;

		return glm::uvec3(i, j, k);
	}

	BoundingBox<glm::dvec3> DensityMap::CalculateSubDomain(const glm::uvec3& index) const
	{
		const glm::dvec3 origin = m_Domain.min + ((glm::dvec3)index * m_CellSize);
		BoundingBox<glm::dvec3> box;
		box.min = origin;
		box.max = origin + m_CellSize;
		return box;
	}

	BoundingBox<glm::dvec3> DensityMap::CalculateSubDomain(const unsigned int index) const
	{
		return CalculateSubDomain(SingleToMultiIndex(index));
	}

	std::array<double, 32> DensityMap::ShapeFunction(const glm::dvec3& xi, std::array<std::array<double, 3>, 32>* gradient)
	{
		auto res = std::array<double, 32>{0.0};

		auto x = xi[0];
		auto y = xi[1];
		auto z = xi[2];

		auto x2 = x * x;
		auto y2 = y * y;
		auto z2 = z * z;

		auto _1mx = 1.0 - x;
		auto _1my = 1.0 - y;
		auto _1mz = 1.0 - z;

		auto _1px = 1.0 + x;
		auto _1py = 1.0 + y;
		auto _1pz = 1.0 + z;

		auto _1m3x = 1.0 - 3.0 * x;
		auto _1m3y = 1.0 - 3.0 * y;
		auto _1m3z = 1.0 - 3.0 * z;

		auto _1p3x = 1.0 + 3.0 * x;
		auto _1p3y = 1.0 + 3.0 * y;
		auto _1p3z = 1.0 + 3.0 * z;

		auto _1mxt1my = _1mx * _1my;
		auto _1mxt1py = _1mx * _1py;
		auto _1pxt1my = _1px * _1my;
		auto _1pxt1py = _1px * _1py;

		auto _1mxt1mz = _1mx * _1mz;
		auto _1mxt1pz = _1mx * _1pz;
		auto _1pxt1mz = _1px * _1mz;
		auto _1pxt1pz = _1px * _1pz;

		auto _1myt1mz = _1my * _1mz;
		auto _1myt1pz = _1my * _1pz;
		auto _1pyt1mz = _1py * _1mz;
		auto _1pyt1pz = _1py * _1pz;

		auto _1mx2 = 1.0 - x2;
		auto _1my2 = 1.0 - y2;
		auto _1mz2 = 1.0 - z2;

		// Corner nodes.
		auto fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
		res[0] = fac * _1mxt1my * _1mz;
		res[1] = fac * _1pxt1my * _1mz;
		res[2] = fac * _1mxt1py * _1mz;
		res[3] = fac * _1pxt1py * _1mz;
		res[4] = fac * _1mxt1my * _1pz;
		res[5] = fac * _1pxt1my * _1pz;
		res[6] = fac * _1mxt1py * _1pz;
		res[7] = fac * _1pxt1py * _1pz;

		// Edge nodes.
		fac = 9.0 / 64.0 * _1mx2;
		auto fact1m3x = fac * _1m3x;
		auto fact1p3x = fac * _1p3x;
		res[8] = fact1m3x * _1myt1mz;
		res[9] = fact1p3x * _1myt1mz;
		res[10] = fact1m3x * _1myt1pz;
		res[11] = fact1p3x * _1myt1pz;
		res[12] = fact1m3x * _1pyt1mz;
		res[13] = fact1p3x * _1pyt1mz;
		res[14] = fact1m3x * _1pyt1pz;
		res[15] = fact1p3x * _1pyt1pz;

		fac = 9.0 / 64.0 * _1my2;
		auto fact1m3y = fac * _1m3y;
		auto fact1p3y = fac * _1p3y;
		res[16] = fact1m3y * _1mxt1mz;
		res[17] = fact1p3y * _1mxt1mz;
		res[18] = fact1m3y * _1pxt1mz;
		res[19] = fact1p3y * _1pxt1mz;
		res[20] = fact1m3y * _1mxt1pz;
		res[21] = fact1p3y * _1mxt1pz;
		res[22] = fact1m3y * _1pxt1pz;
		res[23] = fact1p3y * _1pxt1pz;

		fac = 9.0 / 64.0 * _1mz2;
		auto fact1m3z = fac * _1m3z;
		auto fact1p3z = fac * _1p3z;
		res[24] = fact1m3z * _1mxt1my;
		res[25] = fact1p3z * _1mxt1my;
		res[26] = fact1m3z * _1mxt1py;
		res[27] = fact1p3z * _1mxt1py;
		res[28] = fact1m3z * _1pxt1my;
		res[29] = fact1p3z * _1pxt1my;
		res[30] = fact1m3z * _1pxt1py;
		res[31] = fact1p3z * _1pxt1py;

		if (gradient) {
			auto& dN = *gradient;

			auto _9t3x2py2pz2m19 = 9.0 * (3.0 * x2 + y2 + z2) - 19.0;
			auto _9tx2p3y2pz2m19 = 9.0 * (x2 + 3.0 * y2 + z2) - 19.0;
			auto _9tx2py2p3z2m19 = 9.0 * (x2 + y2 + 3.0 * z2) - 19.0;
			auto _18x = 18.0 * x;
			auto _18y = 18.0 * y;
			auto _18z = 18.0 * z;

			auto _3m9x2 = 3.0 - 9.0 * x2;
			auto _3m9y2 = 3.0 - 9.0 * y2;
			auto _3m9z2 = 3.0 - 9.0 * z2;

			auto _2x = 2.0 * x;
			auto _2y = 2.0 * y;
			auto _2z = 2.0 * z;

			auto _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
			auto _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
			auto _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
			auto _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
			auto _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
			auto _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

			dN[0][0] = _18xm9t3x2py2pz2m19 * _1myt1mz;
			dN[0][1] = _1mxt1mz * _18ym9tx2p3y2pz2m19;
			dN[0][2] = _1mxt1my * _18zm9tx2py2p3z2m19;
			dN[1][0] = _18xp9t3x2py2pz2m19 * _1myt1mz;
			dN[1][1] = _1pxt1mz * _18ym9tx2p3y2pz2m19;
			dN[1][2] = _1pxt1my * _18zm9tx2py2p3z2m19;
			dN[2][0] = _18xm9t3x2py2pz2m19 * _1pyt1mz;
			dN[2][1] = _1mxt1mz * _18yp9tx2p3y2pz2m19;
			dN[2][2] = _1mxt1py * _18zm9tx2py2p3z2m19;
			dN[3][0] = _18xp9t3x2py2pz2m19 * _1pyt1mz;
			dN[3][1] = _1pxt1mz * _18yp9tx2p3y2pz2m19;
			dN[3][2] = _1pxt1py * _18zm9tx2py2p3z2m19;
			dN[4][0] = _18xm9t3x2py2pz2m19 * _1myt1pz;
			dN[4][1] = _1mxt1pz * _18ym9tx2p3y2pz2m19;
			dN[4][2] = _1mxt1my * _18zp9tx2py2p3z2m19;
			dN[5][0] = _18xp9t3x2py2pz2m19 * _1myt1pz;
			dN[5][1] = _1pxt1pz * _18ym9tx2p3y2pz2m19;
			dN[5][2] = _1pxt1my * _18zp9tx2py2p3z2m19;
			dN[6][0] = _18xm9t3x2py2pz2m19 * _1pyt1pz;
			dN[6][1] = _1mxt1pz * _18yp9tx2p3y2pz2m19;
			dN[6][2] = _1mxt1py * _18zp9tx2py2p3z2m19;
			dN[7][0] = _18xp9t3x2py2pz2m19 * _1pyt1pz;
			dN[7][1] = _1pxt1pz * _18yp9tx2p3y2pz2m19;
			dN[7][2] = _1pxt1py * _18zp9tx2py2p3z2m19;

			dN[0][0] /= 64.0;
			dN[0][1] /= 64.0;
			dN[0][2] /= 64.0;
			dN[1][0] /= 64.0;
			dN[1][1] /= 64.0;
			dN[1][2] /= 64.0;
			dN[2][0] /= 64.0;
			dN[2][1] /= 64.0;
			dN[2][2] /= 64.0;
			dN[3][0] /= 64.0;
			dN[3][1] /= 64.0;
			dN[3][2] /= 64.0;
			dN[4][0] /= 64.0;
			dN[4][1] /= 64.0;
			dN[4][2] /= 64.0;
			dN[5][0] /= 64.0;
			dN[5][1] /= 64.0;
			dN[5][2] /= 64.0;
			dN[6][0] /= 64.0;
			dN[6][1] /= 64.0;
			dN[6][2] /= 64.0;
			dN[7][0] /= 64.0;
			dN[7][1] /= 64.0;
			dN[7][2] /= 64.0;

			auto _m3m9x2m2x = -_3m9x2 - _2x;
			auto _p3m9x2m2x = _3m9x2 - _2x;
			auto _1mx2t1m3x = _1mx2 * _1m3x;
			auto _1mx2t1p3x = _1mx2 * _1p3x;
			dN[8][0] = _m3m9x2m2x * _1myt1mz, dN[8][1] = -_1mx2t1m3x * _1mz, dN[8][2] = -_1mx2t1m3x * _1my;
			dN[9][0] = _p3m9x2m2x * _1myt1mz, dN[9][1] = -_1mx2t1p3x * _1mz, dN[9][2] = -_1mx2t1p3x * _1my;
			dN[10][0] = _m3m9x2m2x * _1myt1pz, dN[10][1] = -_1mx2t1m3x * _1pz, dN[10][2] = _1mx2t1m3x * _1my;
			dN[11][0] = _p3m9x2m2x * _1myt1pz, dN[11][1] = -_1mx2t1p3x * _1pz, dN[11][2] = _1mx2t1p3x * _1my;
			dN[12][0] = _m3m9x2m2x * _1pyt1mz, dN[12][1] = _1mx2t1m3x * _1mz, dN[12][2] = -_1mx2t1m3x * _1py;
			dN[13][0] = _p3m9x2m2x * _1pyt1mz, dN[13][1] = _1mx2t1p3x * _1mz, dN[13][2] = -_1mx2t1p3x * _1py;
			dN[14][0] = _m3m9x2m2x * _1pyt1pz, dN[14][1] = _1mx2t1m3x * _1pz, dN[14][2] = _1mx2t1m3x * _1py;
			dN[15][0] = _p3m9x2m2x * _1pyt1pz, dN[15][1] = _1mx2t1p3x * _1pz, dN[15][2] = _1mx2t1p3x * _1py;

			auto _m3m9y2m2y = -_3m9y2 - _2y;
			auto _p3m9y2m2y = _3m9y2 - _2y;
			auto _1my2t1m3y = _1my2 * _1m3y;
			auto _1my2t1p3y = _1my2 * _1p3y;
			dN[16][0] = -_1my2t1m3y * _1mz, dN[16][1] = _m3m9y2m2y * _1mxt1mz, dN[16][2] = -_1my2t1m3y * _1mx;
			dN[17][0] = -_1my2t1p3y * _1mz, dN[17][1] = _p3m9y2m2y * _1mxt1mz, dN[17][2] = -_1my2t1p3y * _1mx;
			dN[18][0] = _1my2t1m3y * _1mz, dN[18][1] = _m3m9y2m2y * _1pxt1mz, dN[18][2] = -_1my2t1m3y * _1px;
			dN[19][0] = _1my2t1p3y * _1mz, dN[19][1] = _p3m9y2m2y * _1pxt1mz, dN[19][2] = -_1my2t1p3y * _1px;
			dN[20][0] = -_1my2t1m3y * _1pz, dN[20][1] = _m3m9y2m2y * _1mxt1pz, dN[20][2] = _1my2t1m3y * _1mx;
			dN[21][0] = -_1my2t1p3y * _1pz, dN[21][1] = _p3m9y2m2y * _1mxt1pz, dN[21][2] = _1my2t1p3y * _1mx;
			dN[22][0] = _1my2t1m3y * _1pz, dN[22][1] = _m3m9y2m2y * _1pxt1pz, dN[22][2] = _1my2t1m3y * _1px;
			dN[23][0] = _1my2t1p3y * _1pz, dN[23][1] = _p3m9y2m2y * _1pxt1pz, dN[23][2] = _1my2t1p3y * _1px;

			auto _m3m9z2m2z = -_3m9z2 - _2z;
			auto _p3m9z2m2z = _3m9z2 - _2z;
			auto _1mz2t1m3z = _1mz2 * _1m3z;
			auto _1mz2t1p3z = _1mz2 * _1p3z;
			dN[24][0] = -_1mz2t1m3z * _1my, dN[24][1] = -_1mz2t1m3z * _1mx, dN[24][2] = _m3m9z2m2z * _1mxt1my;
			dN[25][0] = -_1mz2t1p3z * _1my, dN[25][1] = -_1mz2t1p3z * _1mx, dN[25][2] = _p3m9z2m2z * _1mxt1my;
			dN[26][0] = -_1mz2t1m3z * _1py, dN[26][1] = _1mz2t1m3z * _1mx, dN[26][2] = _m3m9z2m2z * _1mxt1py;
			dN[27][0] = -_1mz2t1p3z * _1py, dN[27][1] = _1mz2t1p3z * _1mx, dN[27][2] = _p3m9z2m2z * _1mxt1py;
			dN[28][0] = _1mz2t1m3z * _1my, dN[28][1] = -_1mz2t1m3z * _1px, dN[28][2] = _m3m9z2m2z * _1pxt1my;
			dN[29][0] = _1mz2t1p3z * _1my, dN[29][1] = -_1mz2t1p3z * _1px, dN[29][2] = _p3m9z2m2z * _1pxt1my;
			dN[30][0] = _1mz2t1m3z * _1py, dN[30][1] = _1mz2t1m3z * _1px, dN[30][2] = _m3m9z2m2z * _1pxt1py;
			dN[31][0] = _1mz2t1p3z * _1py, dN[31][1] = _1mz2t1p3z * _1px, dN[31][2] = _p3m9z2m2z * _1pxt1py;

			constexpr auto rfe = 9.0 / 64.0;
			dN[31][0] *= rfe;
			dN[31][1] *= rfe;
			dN[31][2] *= rfe;
			dN[30][0] *= rfe;
			dN[30][1] *= rfe;
			dN[30][2] *= rfe;
			dN[29][0] *= rfe;
			dN[29][1] *= rfe;
			dN[29][2] *= rfe;
			dN[28][0] *= rfe;
			dN[28][1] *= rfe;
			dN[28][2] *= rfe;
			dN[27][0] *= rfe;
			dN[27][1] *= rfe;
			dN[27][2] *= rfe;
			dN[26][0] *= rfe;
			dN[26][1] *= rfe;
			dN[26][2] *= rfe;
			dN[25][0] *= rfe;
			dN[25][1] *= rfe;
			dN[25][2] *= rfe;
			dN[24][0] *= rfe;
			dN[24][1] *= rfe;
			dN[24][2] *= rfe;
			dN[23][0] *= rfe;
			dN[23][1] *= rfe;
			dN[23][2] *= rfe;
			dN[22][0] *= rfe;
			dN[22][1] *= rfe;
			dN[22][2] *= rfe;
			dN[21][0] *= rfe;
			dN[21][1] *= rfe;
			dN[21][2] *= rfe;
			dN[20][0] *= rfe;
			dN[20][1] *= rfe;
			dN[20][2] *= rfe;
			dN[19][0] *= rfe;
			dN[19][1] *= rfe;
			dN[19][2] *= rfe;
			dN[18][0] *= rfe;
			dN[18][1] *= rfe;
			dN[18][2] *= rfe;
			dN[17][0] *= rfe;
			dN[17][1] *= rfe;
			dN[17][2] *= rfe;
			dN[16][0] *= rfe;
			dN[16][1] *= rfe;
			dN[16][2] *= rfe;
			dN[15][0] *= rfe;
			dN[15][1] *= rfe;
			dN[15][2] *= rfe;
			dN[14][0] *= rfe;
			dN[14][1] *= rfe;
			dN[14][2] *= rfe;
			dN[13][0] *= rfe;
			dN[13][1] *= rfe;
			dN[13][2] *= rfe;
			dN[12][0] *= rfe;
			dN[12][1] *= rfe;
			dN[12][2] *= rfe;
			dN[11][0] *= rfe;
			dN[11][1] *= rfe;
			dN[11][2] *= rfe;
			dN[10][0] *= rfe;
			dN[10][1] *= rfe;
			dN[10][2] *= rfe;
			dN[9][0] *= rfe;
			dN[9][1] *= rfe;
			dN[9][2] *= rfe;
			dN[8][0] *= rfe;
			dN[8][1] *= rfe;
			dN[8][2] *= rfe;
		}

		return res;
	}
}