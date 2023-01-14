#include "pch.h"
#include "DensityMap.cuh"

#include "Compute/ComputeHelper.h"

namespace vfd {
	DensityMap::DensityMap(const BoundingBox<glm::vec3>& domain, glm::uvec3 resolution)
		: m_Domain(domain), m_Resolution(resolution)
	{
		m_CellSize = m_Domain.Diagonal() / static_cast<glm::vec3>(m_Resolution);
		m_CellSizeInverse = 1.0f / m_CellSize;
		m_CellCount = glm::compMul(m_Resolution);
	}

	DensityMap::~DensityMap()
	{
		if(d_DeviceData != nullptr)
		{
			COMPUTE_SAFE(cudaFree(d_DeviceData))
		}
	}

	void DensityMap::AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate)
	{
		glm::uvec3 n = m_Resolution;
		const glm::uvec3 ne = {
			(n[0] + 0) * (n[1] + 1) * (n[2] + 1),
			(n[0] + 1) * (n[1] + 0) * (n[2] + 1),
			(n[0] + 1) * (n[1] + 1) * (n[2] + 0)
		};
		const unsigned int nv = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);
		const unsigned int nec =  glm::compAdd(ne);
		const unsigned int nodeCount = nv + 2 * nec;

		m_Nodes.push_back({});
		auto& coefficients = m_Nodes.back();
		coefficients.resize(nodeCount);

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) nowait
			for (unsigned int l = 0u; l < static_cast<int>(nodeCount); ++l) {
				glm::vec3 x = IndexToNodePosition(l);
				float& c = coefficients[l];

				if (!predicate || predicate(x)) {
					c = function(x);
				}
				else {
					c = std::numeric_limits<float>::max();
				}
			}
		}

		m_Cells.push_back({});
		auto& cells = m_Cells.back();
		cells.resize(m_CellCount);

		for (unsigned int l = 0u; l < m_CellCount; ++l)
		{
			const unsigned int k = l / (n[1] * n[0]);
			const unsigned int temp = l % (n[1] * n[0]);
			const unsigned int j = temp / n[0];
			const unsigned int i = temp % n[0];

			const unsigned int nx = n[0];
			const unsigned int ny = n[1];
			const unsigned int nz = n[2];

			auto& cell = cells[l];
			cell[0] = (nx + 1u) * (ny + 1u) * k + (nx + 1u) * j + i;
			cell[1] = (nx + 1u) * (ny + 1u) * k + (nx + 1u) * j + i + 1u;
			cell[2] = (nx + 1u) * (ny + 1u) * k + (nx + 1u) * (j + 1u) + i;
			cell[3] = (nx + 1u) * (ny + 1u) * k + (nx + 1u) * (j + 1u) + i + 1u;
			cell[4] = (nx + 1u) * (ny + 1u) * (k + 1u) + (nx + 1u) * j + i;
			cell[5] = (nx + 1u) * (ny + 1u) * (k + 1u) + (nx + 1u) * j + i + 1u;
			cell[6] = (nx + 1u) * (ny + 1u) * (k + 1u) + (nx + 1u) * (j + 1u) + i;
			cell[7] = (nx + 1u) * (ny + 1u) * (k + 1u) + (nx + 1u) * (j + 1u) + i + 1u;

			auto offset = nv;
			cell[8] = offset + 2u * (nx * (ny + 1u) * k + nx * j + i);
			cell[9] = cell[8] + 1u;
			cell[10] = offset + 2u * (nx * (ny + 1u) * (k + 1u) + nx * j + i);
			cell[11] = cell[10] + 1u;
			cell[12] = offset + 2u * (nx * (ny + 1u) * k + nx * (j + 1u) + i);
			cell[13] = cell[12] + 1u;
			cell[14] = offset + 2u * (nx * (ny + 1u) * (k + 1u) + nx * (j + 1u) + i);
			cell[15] = cell[14] + 1u;

			offset += 2u * ne.x;
			cell[16] = offset + 2u * (ny * (nz + 1u) * i + ny * k + j);
			cell[17] = cell[16] + 1u;
			cell[18] = offset + 2u * (ny * (nz + 1u) * (i + 1u) + ny * k + j);
			cell[19] = cell[18] + 1u;
			cell[20] = offset + 2u * (ny * (nz + 1u) * i + ny * (k + 1u) + j);
			cell[21] = cell[20] + 1u;
			cell[22] = offset + 2u * (ny * (nz + 1u) * (i + 1u) + ny * (k + 1u) + j);
			cell[23] = cell[22] + 1u;

			offset += 2 * ne.y;
			cell[24] = offset + 2u * (nz * (nx + 1u) * j + nz * i + k);
			cell[25] = cell[24] + 1u;
			cell[26] = offset + 2u * (nz * (nx + 1u) * (j + 1u) + nz * i + k);
			cell[27] = cell[26] + 1u;
			cell[28] = offset + 2u * (nz * (nx + 1u) * j + nz * (i + 1u) + k);
			cell[29] = cell[28] + 1u;
			cell[30] = offset + 2u * (nz * (nx + 1u) * (j + 1u) + nz * (i + 1u) + k);
			cell[31] = cell[30] + 1u;
		}

		m_CellMap.push_back({});
		auto& cell_map = m_CellMap.back();
		cell_map.resize(m_CellCount);
		std::iota(cell_map.begin(), cell_map.end(), 0u);

		m_FieldCount++;
	}

	float DensityMap::Interpolate(unsigned int fieldID, const glm::vec3& point, glm::vec3* gradient) const
	{
		if (m_Domain.Contains(point) == false) {
			return std::numeric_limits<float>::max();
		}

		glm::uvec3 multiIndex = (point - m_Domain.min) * m_CellSizeInverse;
		if (multiIndex.x >= m_Resolution.x) {
			multiIndex.x = m_Resolution.x - 1u;
		}
		if (multiIndex.y >= m_Resolution.y) {
			multiIndex.y = m_Resolution.y - 1u;
		}
		if (multiIndex.z >= m_Resolution.z) {
			multiIndex.z = m_Resolution.z - 1u;
		}

		unsigned int i = MultiToSingleIndex(multiIndex);
		unsigned int j = m_CellMap[fieldID][i];
		if (j == std::numeric_limits<unsigned int>::max()) {
			return std::numeric_limits<float>::max();
		}

		BoundingBox<glm::vec3> subDomain = CalculateSubDomain(i);
		glm::vec3 denominator = subDomain.Diagonal();
		glm::vec3 c0 = 2.0f / denominator;
		glm::vec3 c1 = (subDomain.max + subDomain.min) / denominator;
		glm::vec3 xi = c0 * point - c1;

		auto const& cell = m_Cells[fieldID][j];
		if (!gradient)
		{
			float phi = 0.0f;
			auto N = ShapeFunction(xi);
			for (unsigned int j = 0u; j < 32u; ++j)
			{
				unsigned int v = cell[j];
				float c = m_Nodes[fieldID][v];
				if (c == std::numeric_limits<float>::max())
				{
					return std::numeric_limits<float>::max();
				}

				phi += c * N[j];
			}

			return phi;
		}

		std::array<std::array<float, 3>, 32> dN{};
		auto N = ShapeFunction(xi, &dN);

		float phi = 0.0f;
		*gradient = { 0.0f, 0.0f, 0.0f };

		for (unsigned int j = 0; j < 32; ++j)
		{
			unsigned int v = cell[j];
			float c = m_Nodes[fieldID][v];

			if (c == std::numeric_limits<float>::max())
			{
				*gradient = { 0.0f, 0.0f, 0.0f };
				return std::numeric_limits<float>::max();
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
		if(d_DeviceData != nullptr)
		{
			return d_DeviceData;
		}

		// Flatten nodes
		std::vector<float> nodes;
		const auto nodeCount = static_cast<unsigned int>(m_Nodes[0].size());

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
		const auto cellCount = static_cast<unsigned int>(m_Cells[0].size());

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
		const auto cellMapCount = static_cast<unsigned int>(m_CellMap[0].size());

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

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_DeviceData), sizeof(DensityMapDeviceData)))
		COMPUTE_SAFE(cudaMemcpy(d_DeviceData, temp, sizeof(DensityMapDeviceData), cudaMemcpyHostToDevice))

		delete temp;
		return d_DeviceData;
	}

	const BoundingBox<glm::vec3>& DensityMap::GetBounds() const
	{
		return m_Domain;
	}

	glm::vec3 DensityMap::IndexToNodePosition(unsigned int i) const
	{
		glm::vec3 result;
		glm::vec3 index;

		unsigned int nv = (m_Resolution.x + 1u) * (m_Resolution.y + 1u) * (m_Resolution.z + 1u);

		glm::ivec3 ne = {
			 (m_Resolution.x + 0u) * (m_Resolution.y + 1u) * (m_Resolution.z + 1u),
			 (m_Resolution.x + 1u) * (m_Resolution.y + 0u) * (m_Resolution.z + 1u),
			 (m_Resolution.x + 1u) * (m_Resolution.y + 1u) * (m_Resolution.z + 0u)
		};

		if (i < nv)
		{
			index.z = i / ((m_Resolution.y + 1u) * (m_Resolution.x + 1u));
			unsigned int temp = i % ((m_Resolution.y + 1u) * (m_Resolution.x + 1u));
			index.y = temp / (m_Resolution.x + 1u);
			index.x = temp % (m_Resolution.x + 1u);

			result = m_Domain.min +  m_CellSize * index;
		}
		else if (i < nv + 2u * ne.x)
		{
			i -= nv;
			unsigned int e_ind = i / 2u;
			index.z = e_ind / ((m_Resolution.y + 1u) * m_Resolution.x);
			unsigned int temp = e_ind % ((m_Resolution.y + 1u) * m_Resolution.x);
			index.y = temp / m_Resolution.x;
			index.x = temp % static_cast<unsigned>(m_Resolution.x);

			result = m_Domain.min + m_CellSize * index;
			result.x += (1.0f + i % 2u) / 3.0f * m_CellSize.x;
		}
		else if (i < nv + 2 * (ne.x + ne.y))
		{
			i -= nv + 2u * ne.x;
			unsigned int e_ind = i / 2u;
			index.x = e_ind / ((m_Resolution.z + 1u) * m_Resolution.y);
			unsigned int temp = e_ind % ((m_Resolution.z + 1u) * m_Resolution.y);
			index.z = temp / m_Resolution.y;
			index.y = temp % static_cast<unsigned>(m_Resolution.y);

			result = m_Domain.min + m_CellSize * index;
			result.y += (1.0f + i % 2u) / 3.0f * m_CellSize.y;
		}
		else
		{
			i -= nv + 2u * (ne.x + ne.y);
			unsigned int e_ind = i / 2u;
			index.y = e_ind / ((m_Resolution.x + 1u) * m_Resolution.z);
			unsigned int temp = e_ind % ((m_Resolution.x + 1u) * m_Resolution.z);
			index.x = temp / m_Resolution.z;
			index.z = temp % static_cast<unsigned>(m_Resolution.z);

			result = m_Domain.min + m_CellSize * index;
			result.z += (1.0f + i % 2u) / 3.0f * m_CellSize.z;
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
		const unsigned int k = index / n01;
		const unsigned int temp = index % n01;
		const float j = temp / m_Resolution.x;
		const float i = temp % m_Resolution.x;

		return { i, j, k };
	}

	BoundingBox<glm::vec3> DensityMap::CalculateSubDomain(const glm::uvec3& index) const
	{
		const glm::vec3 origin = m_Domain.min + static_cast<glm::vec3>(index) * m_CellSize;
		const BoundingBox<glm::vec3> box(origin, origin + m_CellSize);
		return box;
	}

	BoundingBox<glm::vec3> DensityMap::CalculateSubDomain(const unsigned int index) const
	{
		return CalculateSubDomain(SingleToMultiIndex(index));
	}

	std::array<float, 32> DensityMap::ShapeFunction(const glm::vec3& xi, std::array<std::array<float, 3>, 32>* gradient)
	{
		auto res = std::array<float, 32>{0.0};

		const float x = xi[0];
		const float y = xi[1];
		const float z = xi[2];

		const float x2 = x * x;
		const float y2 = y * y;
		const float z2 = z * z;

		const float _1mx = 1.0f - x;
		const float _1my = 1.0f - y;
		const float _1mz = 1.0f - z;

		const float _1px = 1.0f + x;
		const float _1py = 1.0f + y;
		const float _1pz = 1.0f + z;

		const float _1m3x = 1.0f - 3.0f * x;
		const float _1m3y = 1.0f - 3.0f * y;
		const float _1m3z = 1.0f - 3.0f * z;

		const float _1p3x = 1.0f + 3.0f * x;
		const float _1p3y = 1.0f + 3.0f * y;
		const float _1p3z = 1.0f + 3.0f * z;

		const float _1mxt1my = _1mx * _1my;
		const float _1mxt1py = _1mx * _1py;
		const float _1pxt1my = _1px * _1my;
		const float _1pxt1py = _1px * _1py;

		const float _1mxt1mz = _1mx * _1mz;
		const float _1mxt1pz = _1mx * _1pz;
		const float _1pxt1mz = _1px * _1mz;
		const float _1pxt1pz = _1px * _1pz;

		const float _1myt1mz = _1my * _1mz;
		const float _1myt1pz = _1my * _1pz;
		const float _1pyt1mz = _1py * _1mz;
		const float _1pyt1pz = _1py * _1pz;

		const float _1mx2 = 1.0f - x2;
		const float _1my2 = 1.0f - y2;
		const float _1mz2 = 1.0f - z2;

		// Corner nodes.
		float fac = 1.0f / 64.0f * (9.0f * (x2 + y2 + z2) - 19.0f);
		res[0] = fac * _1mxt1my * _1mz;
		res[1] = fac * _1pxt1my * _1mz;
		res[2] = fac * _1mxt1py * _1mz;
		res[3] = fac * _1pxt1py * _1mz;
		res[4] = fac * _1mxt1my * _1pz;
		res[5] = fac * _1pxt1my * _1pz;
		res[6] = fac * _1mxt1py * _1pz;
		res[7] = fac * _1pxt1py * _1pz;

		// Edge nodes.
		fac = 9.0f / 64.0f * _1mx2;
		const float fact1m3x = fac * _1m3x;
		const float fact1p3x = fac * _1p3x;
		res[8] = fact1m3x * _1myt1mz;
		res[9] = fact1p3x * _1myt1mz;
		res[10] = fact1m3x * _1myt1pz;
		res[11] = fact1p3x * _1myt1pz;
		res[12] = fact1m3x * _1pyt1mz;
		res[13] = fact1p3x * _1pyt1mz;
		res[14] = fact1m3x * _1pyt1pz;
		res[15] = fact1p3x * _1pyt1pz;

		fac = 9.0f / 64.0f * _1my2;
		const float fact1m3y = fac * _1m3y;
		const float fact1p3y = fac * _1p3y;
		res[16] = fact1m3y * _1mxt1mz;
		res[17] = fact1p3y * _1mxt1mz;
		res[18] = fact1m3y * _1pxt1mz;
		res[19] = fact1p3y * _1pxt1mz;
		res[20] = fact1m3y * _1mxt1pz;
		res[21] = fact1p3y * _1mxt1pz;
		res[22] = fact1m3y * _1pxt1pz;
		res[23] = fact1p3y * _1pxt1pz;

		fac = 9.0f / 64.0f * _1mz2;
		const float fact1m3z = fac * _1m3z;
		const float fact1p3z = fac * _1p3z;
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

			const float _9t3x2py2pz2m19 = 9.0f * (3.0f * x2 + y2 + z2) - 19.0f;
			const float _9tx2p3y2pz2m19 = 9.0f * (x2 + 3.0f * y2 + z2) - 19.0f;
			const float _9tx2py2p3z2m19 = 9.0f * (x2 + y2 + 3.0f * z2) - 19.0f;
			const float _18x = 18.0f * x;
			const float _18y = 18.0f * y;
			const float _18z = 18.0f * z;

			const float _3m9x2 = 3.0f - 9.0f * x2;
			const float _3m9y2 = 3.0f - 9.0f * y2;
			const float _3m9z2 = 3.0f - 9.0f * z2;

			const float _2x = 2.0f * x;
			const float _2y = 2.0f * y;
			const float _2z = 2.0f * z;

			const float _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
			const float _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
			const float _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
			const float _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
			const float _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
			const float _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

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

			dN[0][0] /= 64.0f;
			dN[0][1] /= 64.0f;
			dN[0][2] /= 64.0f;
			dN[1][0] /= 64.0f;
			dN[1][1] /= 64.0f;
			dN[1][2] /= 64.0f;
			dN[2][0] /= 64.0f;
			dN[2][1] /= 64.0f;
			dN[2][2] /= 64.0f;
			dN[3][0] /= 64.0f;
			dN[3][1] /= 64.0f;
			dN[3][2] /= 64.0f;
			dN[4][0] /= 64.0f;
			dN[4][1] /= 64.0f;
			dN[4][2] /= 64.0f;
			dN[5][0] /= 64.0f;
			dN[5][1] /= 64.0f;
			dN[5][2] /= 64.0f;
			dN[6][0] /= 64.0f;
			dN[6][1] /= 64.0f;
			dN[6][2] /= 64.0f;
			dN[7][0] /= 64.0f;
			dN[7][1] /= 64.0f;
			dN[7][2] /= 64.0f;

			const float _m3m9x2m2x = -_3m9x2 - _2x;
			const float _p3m9x2m2x = _3m9x2 - _2x;
			const float _1mx2t1m3x = _1mx2 * _1m3x;
			const float _1mx2t1p3x = _1mx2 * _1p3x;

			dN[8][0] = _m3m9x2m2x * _1myt1mz;
			dN[8][1] = -_1mx2t1m3x * _1mz;
			dN[8][2] = -_1mx2t1m3x * _1my;
			dN[9][0] = _p3m9x2m2x * _1myt1mz;
			dN[9][1] = -_1mx2t1p3x * _1mz;
			dN[9][2] = -_1mx2t1p3x * _1my;
			dN[10][0] = _m3m9x2m2x * _1myt1pz;
			dN[10][1] = -_1mx2t1m3x * _1pz;
			dN[10][2] = _1mx2t1m3x * _1my;
			dN[11][0] = _p3m9x2m2x * _1myt1pz;
			dN[11][1] = -_1mx2t1p3x * _1pz;
			dN[11][2] = _1mx2t1p3x * _1my;
			dN[12][0] = _m3m9x2m2x * _1pyt1mz;
			dN[12][1] = _1mx2t1m3x * _1mz;
			dN[12][2] = -_1mx2t1m3x * _1py;
			dN[13][0] = _p3m9x2m2x * _1pyt1mz;
			dN[13][1] = _1mx2t1p3x * _1mz;
			dN[13][2] = -_1mx2t1p3x * _1py;
			dN[14][0] = _m3m9x2m2x * _1pyt1pz;
			dN[14][1] = _1mx2t1m3x * _1pz;
			dN[14][2] = _1mx2t1m3x * _1py;
			dN[15][0] = _p3m9x2m2x * _1pyt1pz;
			dN[15][1] = _1mx2t1p3x * _1pz;
			dN[15][2] = _1mx2t1p3x * _1py;

			const float _m3m9y2m2y = -_3m9y2 - _2y;
			const float _p3m9y2m2y = _3m9y2 - _2y;
			const float _1my2t1m3y = _1my2 * _1m3y;
			const float _1my2t1p3y = _1my2 * _1p3y;

			dN[16][0] = -_1my2t1m3y * _1mz;
			dN[16][1] = _m3m9y2m2y * _1mxt1mz;
			dN[16][2] = -_1my2t1m3y * _1mx;
			dN[17][0] = -_1my2t1p3y * _1mz;
			dN[17][1] = _p3m9y2m2y * _1mxt1mz;
			dN[17][2] = -_1my2t1p3y * _1mx;
			dN[18][0] = _1my2t1m3y * _1mz;
			dN[18][1] = _m3m9y2m2y * _1pxt1mz;
			dN[18][2] = -_1my2t1m3y * _1px;
			dN[19][0] = _1my2t1p3y * _1mz;
			dN[19][1] = _p3m9y2m2y * _1pxt1mz;
			dN[19][2] = -_1my2t1p3y * _1px;
			dN[20][0] = -_1my2t1m3y * _1pz;
			dN[20][1] = _m3m9y2m2y * _1mxt1pz;
			dN[20][2] = _1my2t1m3y * _1mx;
			dN[21][0] = -_1my2t1p3y * _1pz;
			dN[21][1] = _p3m9y2m2y * _1mxt1pz;
			dN[21][2] = _1my2t1p3y * _1mx;
			dN[22][0] = _1my2t1m3y * _1pz;
			dN[22][1] = _m3m9y2m2y * _1pxt1pz;
			dN[22][2] = _1my2t1m3y * _1px;
			dN[23][0] = _1my2t1p3y * _1pz;
			dN[23][1] = _p3m9y2m2y * _1pxt1pz;
			dN[23][2] = _1my2t1p3y * _1px;

			const float _m3m9z2m2z = -_3m9z2 - _2z;
			const float _p3m9z2m2z = _3m9z2 - _2z;
			const float _1mz2t1m3z = _1mz2 * _1m3z;
			const float _1mz2t1p3z = _1mz2 * _1p3z;

			dN[24][0] = -_1mz2t1m3z * _1my;
			dN[24][1] = -_1mz2t1m3z * _1mx;
			dN[24][2] = _m3m9z2m2z * _1mxt1my;
			dN[25][0] = -_1mz2t1p3z * _1my;
			dN[25][1] = -_1mz2t1p3z * _1mx;
			dN[25][2] = _p3m9z2m2z * _1mxt1my;
			dN[26][0] = -_1mz2t1m3z * _1py;
			dN[26][1] = _1mz2t1m3z * _1mx;
			dN[26][2] = _m3m9z2m2z * _1mxt1py;
			dN[27][0] = -_1mz2t1p3z * _1py;
			dN[27][1] = _1mz2t1p3z * _1mx;
			dN[27][2] = _p3m9z2m2z * _1mxt1py;
			dN[28][0] = _1mz2t1m3z * _1my;
			dN[28][1] = -_1mz2t1m3z * _1px;
			dN[28][2] = _m3m9z2m2z * _1pxt1my;
			dN[29][0] = _1mz2t1p3z * _1my;
			dN[29][1] = -_1mz2t1p3z * _1px;
			dN[29][2] = _p3m9z2m2z * _1pxt1my;
			dN[30][0] = _1mz2t1m3z * _1py;
			dN[30][1] = _1mz2t1m3z * _1px;
			dN[30][2] = _m3m9z2m2z * _1pxt1py;
			dN[31][0] = _1mz2t1p3z * _1py;
			dN[31][1] = _1mz2t1p3z * _1px;
			dN[31][2] = _p3m9z2m2z * _1pxt1py;

			constexpr float rfe = 9.0f / 64.0f;
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
			dN[9 ][0] *= rfe;
			dN[9 ][1] *= rfe;
			dN[9 ][2] *= rfe;
			dN[8 ][0] *= rfe;
			dN[8 ][1] *= rfe;
			dN[8 ][2] *= rfe;
		}

		return res;
	}
}