#include "pch.h"
#include "SDF.h"

#include "MeshDistance.h"

namespace fe {
	SDF::SDF(const BoundingBox& domain, glm::ivec3 resolution)
	: m_Resolution(resolution), m_Domain(domain), m_CellCount(0u)
	{
		m_CellSize = (glm::vec3)m_Domain.Diagonal() / (glm::vec3)m_Resolution;
		m_CellSizeInverse = 1.0 / m_CellSize;
		m_CellCount = m_Resolution.x * m_Resolution.y * m_Resolution.z;
	}

	SDF::SDF(const EdgeMesh& mesh, const BoundingBox& bounds, const glm::uvec3& resolution, const bool inverted)
		: m_Resolution(resolution)
	{
		MeshDistance distance(mesh);

		m_Domain.Extend(bounds.min);
		m_Domain.Extend(bounds.max);
		m_Domain.max += 0.001f * glm::sqrt(glm::dot(m_Domain.Diagonal(), m_Domain.Diagonal()));
		m_Domain.min -= 0.001f * glm::sqrt(glm::dot(m_Domain.Diagonal(), m_Domain.Diagonal()));

		m_CellSize = (glm::vec3)m_Domain.Diagonal() / (glm::vec3)m_Resolution;
		m_CellSizeInverse = 1.0 / m_CellSize;
		m_CellCount = m_Resolution.x * m_Resolution.y * m_Resolution.z;

		float factor = 1.0;
		if (inverted) {
			factor = -1.0;
		}

		const ContinuousFunction function = [&distance, &factor](const glm::vec3& xi) {
			return factor * distance.SignedDistanceCached(xi);
		};

		AddFunction(function);
	}



	SDF::SDF(const std::string& filepath)
	{
		auto in = std::ifstream(filepath, std::ios::binary);

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
		read(*in.rdbuf(), m_FieldCount);

		auto n_nodes = std::size_t{};
		read(*in.rdbuf(), n_nodes);
		m_Nodes.resize(n_nodes);

		for (auto& nodes : m_Nodes)
		{
			read(*in.rdbuf(), n_nodes);
			nodes.resize(n_nodes);
			for (auto& node : nodes)
			{
				read(*in.rdbuf(), node);
			}
		}

		auto n_cells = std::size_t{};
		read(*in.rdbuf(), n_cells);
		m_Cells.resize(n_cells);
		for (auto& cells : m_Cells)
		{
			read(*in.rdbuf(), n_cells);
			cells.resize(n_cells);
			for (auto& cell : cells)
			{
				read(*in.rdbuf(), cell);
			}
		}

		auto n_cell_maps = std::size_t{};
		read(*in.rdbuf(), n_cell_maps);
		m_CellMap.resize(n_cell_maps);
		for (auto& cell_maps : m_CellMap)
		{
			read(*in.rdbuf(), n_cell_maps);
			cell_maps.resize(n_cell_maps);
			for (auto& cell_map : cell_maps)
			{
				read(*in.rdbuf(), cell_map);
			}
		}

		in.close();
	}

	uint32_t SDF::AddFunction(const ContinuousFunction& function, const SamplePredicate& predicate)
	{
		const uint32_t nv = (m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 1);

		const glm::ivec3 ne = {
			(m_Resolution.x + 0) * (m_Resolution.y + 1) * (m_Resolution.z + 1),
			(m_Resolution.x + 1) * (m_Resolution.y + 0) * (m_Resolution.z + 1),
			(m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 0)
		};

		const uint32_t nes = ne.x + ne.y + ne.z;
		const uint32_t nodeCount = nv + 2 * nes;

		m_Nodes.push_back({});
		auto& coeffs = m_Nodes.back();
		coeffs.resize(nodeCount);

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) nowait
			for (int l = 0; l < static_cast<int>(nodeCount); ++l) {
				glm::vec3 x = IndexToNodePosition(l);
				double& c = coeffs[l];

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

		for (uint32_t l = 0; l < m_CellCount; ++l)
		{
			const uint32_t k = l / (m_Resolution[1] * m_Resolution[0]);
			const uint32_t temp = l % (m_Resolution[1] * m_Resolution[0]);
			const uint32_t j = temp / m_Resolution[0];
			const uint32_t i = temp % m_Resolution[0];

			const uint32_t nx = m_Resolution[0];
			const uint32_t ny = m_Resolution[1];
			const uint32_t nz = m_Resolution[2];

			auto& cell = cells[l];
			cell[0] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i;
			cell[1] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i + 1;
			cell[2] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i;
			cell[3] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i + 1;
			cell[4] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i;
			cell[5] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i + 1;
			cell[6] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i;
			cell[7] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i + 1;

			uint32_t offset = nv;
			cell[8] = offset + 2 * (nx * (ny + 1) * k + nx * j + i);
			cell[9] = cell[8] + 1;
			cell[10] = offset + 2 * (nx * (ny + 1) * (k + 1) + nx * j + i);
			cell[11] = cell[10] + 1;
			cell[12] = offset + 2 * (nx * (ny + 1) * k + nx * (j + 1) + i);
			cell[13] = cell[12] + 1;
			cell[14] = offset + 2 * (nx * (ny + 1) * (k + 1) + nx * (j + 1) + i);
			cell[15] = cell[14] + 1;

			offset += 2 * ne.x;
			cell[16] = offset + 2 * (ny * (nz + 1) * i + ny * k + j);
			cell[17] = cell[16] + 1;
			cell[18] = offset + 2 * (ny * (nz + 1) * (i + 1) + ny * k + j);
			cell[19] = cell[18] + 1;
			cell[20] = offset + 2 * (ny * (nz + 1) * i + ny * (k + 1) + j);
			cell[21] = cell[20] + 1;
			cell[22] = offset + 2 * (ny * (nz + 1) * (i + 1) + ny * (k + 1) + j);
			cell[23] = cell[22] + 1;

			offset += 2 * ne.y;
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

		return m_FieldCount++;
	}

	glm::vec3 SDF::IndexToNodePosition(uint32_t i) const
	{
		glm::vec3 result;
		glm::vec3 index;

		uint32_t nv = (m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 1);

		glm::ivec3 ne = {
			 (m_Resolution.x + 0) * (m_Resolution.y + 1) * (m_Resolution.z + 1),
			 (m_Resolution.x + 1) * (m_Resolution.y + 0) * (m_Resolution.z + 1),
			 (m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 0)
		};

		if (i < nv)
		{
			index.z = i / (uint32_t)((m_Resolution.y + 1) * (m_Resolution.x + 1));
			uint32_t temp = i % (uint32_t)((m_Resolution.y + 1) * (m_Resolution.x + 1));
			index.y = temp / (m_Resolution.x + 1);
			index.x = temp % (uint32_t)(m_Resolution.x + 1);

			result = (glm::vec3)m_Domain.min + (glm::vec3)m_CellSize * index;
		}
		else if (i < nv + 2 * ne.x)
		{
			i -= nv;
			uint32_t e_ind = i / 2;
			index.z = e_ind / ((m_Resolution.y + 1) * m_Resolution.x);
			uint32_t temp = e_ind % (uint32_t)((m_Resolution.y + 1) * m_Resolution.x);
			index.y = temp / m_Resolution.x;
			index.x = temp % (uint32_t)m_Resolution.x;

			result = (glm::vec3)m_Domain.min + (glm::vec3)m_CellSize * index;
			result.x += (1.0 + i % 2) / 3.0 * m_CellSize.x;
		}
		else if (i < nv + 2 * (ne.x + ne.y))
		{
			i -= (nv + 2 * ne.x);
			uint32_t e_ind = i / 2;
			index.x = e_ind / ((m_Resolution.z + 1) * m_Resolution.y);
			uint32_t temp = e_ind % (uint32_t)((m_Resolution.z + 1) * m_Resolution.y);
			index.z = temp / m_Resolution.y;
			index.y = temp % (uint32_t)m_Resolution.y;

			result = (glm::vec3)m_Domain.min + (glm::vec3)m_CellSize * index;
			result.y += (1.0 + i % 2) / 3.0 * m_CellSize.y;
		}
		else
		{
			i -= (nv + 2 * (ne.x + ne.y));
			uint32_t e_ind = i / 2;
			index.y = e_ind / ((m_Resolution.x + 1) * m_Resolution.z);
			uint32_t temp = e_ind % (uint32_t)((m_Resolution.x + 1) * m_Resolution.z);
			index.x = temp / m_Resolution.z;
			index.z = temp % (uint32_t)m_Resolution.z;

			result = (glm::vec3)m_Domain.min + (glm::vec3)m_CellSize * index;
			result.z += (1.0 + i % 2) / 3.0 * m_CellSize.z;
		}

		return result;
	}

	float SDF::Interpolate(unsigned int fieldID, const glm::vec3& point, glm::vec3* gradient) const
	{
		if (m_Domain.Contains(point) == false) {
			return std::numeric_limits<float>::max();
		}

		glm::ivec3 multiIndex = (point - (glm::vec3)m_Domain.min) * (glm::vec3)m_CellSizeInverse;
		if (multiIndex.x >= m_Resolution.x) {
			multiIndex.x = m_Resolution.x - 1;
		}
		if (multiIndex.y >= m_Resolution.y) {
			multiIndex.y = m_Resolution.y - 1;
		}
		if (multiIndex.z >= m_Resolution.z) {
			multiIndex.z = m_Resolution.z - 1;
		}

		uint32_t index = MultiToSingleIndex(multiIndex);
		uint32_t index_ = m_CellMap[fieldID][index];
		if (index_ == std::numeric_limits<uint32_t>::max()) {
			return std::numeric_limits<float>::max();
		}

		BoundingBox subDomain = CalculateSubDomain(index);
		index = index_;
		glm::vec3 d = subDomain.Diagonal();
		glm::vec3 denom = (subDomain.max - subDomain.min);
		glm::vec3 c0 = 2.0f / denom;
		glm::vec3 c1 = (glm::vec3)(subDomain.max + subDomain.min) / (denom);
		glm::vec3 xi = (c0 * point - c1);

		auto const& cell = m_Cells[fieldID][index];
		if (!gradient)
		{
			float phi = 0.0;
			auto N = ShapeFunction(xi);
			for (uint32_t j = 0; j < 32; ++j)
			{
				uint32_t v = cell[j];
				float c = m_Nodes[fieldID][v];
				if (c == std::numeric_limits<float>::max())
				{
					return std::numeric_limits<float>::max();
				}

				phi += c * N[j];
			}

			return phi;
		}

		std::array<std::array<double, 3>, 32> dN{};
		auto N = ShapeFunction(xi, &dN);

		float phi = 0.0;
		*gradient = { 0.0, 0.0, 0.0 };

		for (uint32_t j = 0; j < 32; ++j)
		{
			uint32_t v = cell[j];
			float c = m_Nodes[fieldID][v];

			if (c == std::numeric_limits<float>::max())
			{
				*gradient = { 0.0, 0.0, 0.0 };
				return std::numeric_limits<float>::max();
			}

			phi += c * N[j];

			(*gradient).x += c * dN[j][0];
			(*gradient).y += c * dN[j][1];
			(*gradient).z += c * dN[j][2];
		}
		// 	gradient->array() *= c0.array();
		return phi;
	}

	double SDF::Interpolate(unsigned int fieldID, const glm::dvec3& xi, const std::array<unsigned int, 32>& cell, const glm::dvec3& c0, const std::array<double, 32>& N, glm::dvec3* gradient, std::array<std::array<double, 3>, 32>* dN)
	{
		if (!gradient)
		{
			double phi = 0.0;
			for (unsigned int j = 0u; j < 32u; ++j)
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

		double phi = 0.0;
		*gradient = { 0.0, 0.0, 0.0 };
		for (unsigned int j = 0u; j < 32u; ++j)
		{
			unsigned int v = cell[j];
			double c = m_Nodes[fieldID][v];
			if (c == std::numeric_limits<double>::max())
			{
				*gradient = { 0.0, 0.0, 0.0 };;
				return std::numeric_limits<double>::max();
			}
			phi += c * N[j];
			(*gradient).x += c * (*dN)[j][0];
			(*gradient).y += c * (*dN)[j][1];
			(*gradient).z += c * (*dN)[j][2];
		}

		// 	gradient->array() *= c0.array();
		return phi;
	}

	bool SDF::DetermineShapeFunctions(unsigned int fieldID, const glm::dvec3& x, std::array<unsigned int, 32>& cell, glm::dvec3& c0, std::array<double, 32>& N, std::array<std::array<double, 3>, 32>* dN)
	{
		if (m_Domain.Contains(x) == false) {
			return false;
		}

		auto mi = (x - m_Domain.min * m_CellSizeInverse);
		if (mi[0] >= m_Resolution[0])
			mi[0] = m_Resolution[0] - 1;
		if (mi[1] >= m_Resolution[1])
			mi[1] = m_Resolution[1] - 1;
		if (mi[2] >= m_Resolution[2])
			mi[2] = m_Resolution[2] - 1;
		unsigned int i = MultiToSingleIndex(mi);
		unsigned int i_ = m_CellMap[fieldID][i];
		if (i_ == std::numeric_limits<unsigned int>::max())
			return false;

		BoundingBox sd = CalculateSubDomain(i);
		i = i_;
		glm::dvec3 d = sd.Diagonal();
		glm::dvec3 denom = (sd.max - sd.min);
		c0 = glm::dvec3(2.0, 2.0, 2.0) / denom;
		glm::dvec3 c1 = (sd.max + sd.min) / denom;
		glm::dvec3 xi = (c0 * x) - c1;

		cell = m_Cells[fieldID][i];
		N = ShapeFunction(xi, dN);
		return true;
	}

	glm::ivec3 SDF::SingleToMultiIndex(const uint32_t index) const
	{
		const uint32_t n01 = m_Resolution.x * m_Resolution.y;
		uint32_t k = index / n01;
		const uint32_t temp = index % n01;
		uint32_t j = temp / m_Resolution.x;
		uint32_t i = temp % m_Resolution.x;

		return { i, j ,k };
	}

	uint32_t SDF::MultiToSingleIndex(const glm::ivec3& index) const
	{
		return m_Resolution.y * m_Resolution.x * index.z + m_Resolution.x * index.y + index.x;
	}

	BoundingBox SDF::CalculateSubDomain(const glm::vec3& index) const
	{
		const glm::vec3 origin = (glm::vec3)m_Domain.min + index * (glm::vec3)m_CellSize;
		BoundingBox box;
		box.min = origin;
		box.max = origin + (glm::vec3)m_CellSize;
		return box;
	}

	BoundingBox SDF::CalculateSubDomain(const uint32_t index) const
	{
		return CalculateSubDomain(SingleToMultiIndex(index));
	}

	std::array<double, 32> SDF::ShapeFunction(const glm::vec3& xi, std::array<std::array<double, 3>, 32>* gradient)
	{
		auto res = std::array<double, 32>{};

		const double x = xi[0];
		const double y = xi[1];
		const double z = xi[2];
		const double x2 = x * x;
		const double y2 = y * y;
		const double z2 = z * z;
		const double mx = 1.0 - x;
		const double my = 1.0 - y;
		const double mz = 1.0 - z;
		const double mx2 = 1.0 - x2;
		const double my2 = 1.0 - y2;
		const double mz2 = 1.0 - z2;
		const double px = 1.0 + x;
		const double py = 1.0 + y;
		const double pz = 1.0 + z;
		const double mx3 = 1.0 - 3.0 * x;
		const double my3 = 1.0 - 3.0 * y;
		const double mz3 = 1.0 - 3.0 * z;
		const double px3 = 1.0 + 3.0 * x;
		const double py3 = 1.0 + 3.0 * y;
		const double pz3 = 1.0 + 3.0 * z;
		const double mxtmy = mx * my;
		const double mxtpy = mx * py;
		const double pxtmy = px * my;
		const double pxtpy = px * py;
		const double mxtmz = mx * mz;
		const double mxtpz = mx * pz;
		const double pxtmz = px * mz;
		const double pxtpz = px * pz;
		const double mytmz = my * mz;
		const double mytpz = my * pz;
		const double pytmz = py * mz;
		const double pytpz = py * pz;

		// Corners
		double fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
		res[0] = fac * mxtmy * mz;
		res[1] = fac * pxtmy * mz;
		res[2] = fac * mxtpy * mz;
		res[3] = fac * pxtpy * mz;
		res[4] = fac * mxtmy * pz;
		res[5] = fac * pxtmy * pz;
		res[6] = fac * mxtpy * pz;
		res[7] = fac * pxtpy * pz;

		// Edges
		fac = 9.0 / 64.0 * mx2;
		const double factmx3 = fac * mx3;
		const double factpx3 = fac * px3;
		res[8] = factmx3 * mytmz;
		res[9] = factpx3 * mytmz;
		res[10] = factmx3 * mytpz;
		res[11] = factpx3 * mytpz;
		res[12] = factmx3 * pytmz;
		res[13] = factpx3 * pytmz;
		res[14] = factmx3 * pytpz;
		res[15] = factpx3 * pytpz;

		fac = 9.0 / 64.0 * my2;
		const double factmy3 = fac * my3;
		const double factpy3 = fac * py3;
		res[16] = factmy3 * mxtmz;
		res[17] = factpy3 * mxtmz;
		res[18] = factmy3 * pxtmz;
		res[19] = factpy3 * pxtmz;
		res[20] = factmy3 * mxtpz;
		res[21] = factpy3 * mxtpz;
		res[22] = factmy3 * pxtpz;
		res[23] = factpy3 * pxtpz;

		fac = 9.0 / 64.0 * mz2;
		const double factmz3 = fac * mz3;
		const double factpz3 = fac * pz3;
		res[24] = factmz3 * mxtmy;
		res[25] = factpz3 * mxtmy;
		res[26] = factmz3 * mxtpy;
		res[27] = factpz3 * mxtpy;
		res[28] = factmz3 * pxtmy;
		res[29] = factpz3 * pxtmy;
		res[30] = factmz3 * pxtpy;
		res[31] = factpz3 * pxtpy;

		if (gradient) {
			auto& dN = *gradient;

			const double t9x3py2pzy2m19 = 9.0 * (3.0 * x2 + y2 + z2) - 19.0;
			const double t9x2p3y2pz2m19 = 9.0 * (x2 + 3.0 * y2 + z2) - 19.0;
			const double t9x2py2p3z2m19 = 9.0 * (x2 + y2 + 3.0 * z2) - 19.0;
			const double x18 = 18.0 * x;
			const double y18 = 18.0 * y;
			const double z18 = 18.0 * z;
			const double m2x92 = 3.0 - 9.0 * x2;
			const double m2y92 = 3.0 - 9.0 * y2;
			const double m3z92 = 3.0 - 9.0 * z2;
			const double x2 = 2.0 * x;
			const double y2 = 2.0 * y;
			const double z2 = 2.0 * z;
			const double x18xm9t3x2py2pz2m19 = x18 - t9x3py2pzy2m19;
			const double y18xp9t3x2py2pz2m19 = x18 + t9x3py2pzy2m19;
			const double z18ym9tx2p3y2pz2m19 = y18 - t9x2p3y2pz2m19;
			const double x18yp9tx2p3y2pz2m19 = y18 + t9x2p3y2pz2m19;
			const double y18zm9tx2py2p3z2m19 = z18 - t9x2py2p3z2m19;
			const double z18zp9tx2py2p3z2m19 = z18 + t9x2py2p3z2m19;

			dN[0][0] = x18xm9t3x2py2pz2m19 * mytmz;
			dN[0][1] = mxtmz * z18ym9tx2p3y2pz2m19;
			dN[0][2] = mxtmy * y18zm9tx2py2p3z2m19;
			dN[1][0] = y18xp9t3x2py2pz2m19 * mytmz;
			dN[1][1] = pxtmz * z18ym9tx2p3y2pz2m19;
			dN[1][2] = pxtmy * y18zm9tx2py2p3z2m19;
			dN[2][0] = x18xm9t3x2py2pz2m19 * pytmz;
			dN[2][1] = mxtmz * x18yp9tx2p3y2pz2m19;
			dN[2][2] = mxtpy * y18zm9tx2py2p3z2m19;
			dN[3][0] = y18xp9t3x2py2pz2m19 * pytmz;
			dN[3][1] = pxtmz * x18yp9tx2p3y2pz2m19;
			dN[3][2] = pxtpy * y18zm9tx2py2p3z2m19;
			dN[4][0] = x18xm9t3x2py2pz2m19 * mytpz;
			dN[4][1] = mxtpz * z18ym9tx2p3y2pz2m19;
			dN[4][2] = mxtmy * z18zp9tx2py2p3z2m19;
			dN[5][0] = y18xp9t3x2py2pz2m19 * mytpz;
			dN[5][1] = pxtpz * z18ym9tx2p3y2pz2m19;
			dN[5][2] = pxtmy * z18zp9tx2py2p3z2m19;
			dN[6][0] = x18xm9t3x2py2pz2m19 * pytpz;
			dN[6][1] = mxtpz * x18yp9tx2p3y2pz2m19;
			dN[6][2] = mxtpy * z18zp9tx2py2p3z2m19;
			dN[7][0] = y18xp9t3x2py2pz2m19 * pytpz;
			dN[7][1] = pxtpz * x18yp9tx2p3y2pz2m19;
			dN[7][2] = pxtpy * z18zp9tx2py2p3z2m19;

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

			const double m3m9x2m2x = -m2x92 - x2;
			const double p3m9x2m2x = m2x92 - x2;
			const double p1mx2t1m3x = mx2 * mx3;
			const double m1mx2t1p3x = mx2 * px3;
			dN[8][0]  = m3m9x2m2x * mytmz, dN[8][1]  = -p1mx2t1m3x * mz, dN[8][2]  = -p1mx2t1m3x * my;
			dN[9][0]  = p3m9x2m2x * mytmz, dN[9][1]  = -m1mx2t1p3x * mz, dN[9][2]  = -m1mx2t1p3x * my;
			dN[10][0] = m3m9x2m2x * mytpz, dN[10][1] = -p1mx2t1m3x * pz, dN[10][2] =  p1mx2t1m3x * my;
			dN[11][0] = p3m9x2m2x * mytpz, dN[11][1] = -m1mx2t1p3x * pz, dN[11][2] =  m1mx2t1p3x * my;
			dN[12][0] = m3m9x2m2x * pytmz, dN[12][1] =  p1mx2t1m3x * mz, dN[12][2] = -p1mx2t1m3x * py;
			dN[13][0] = p3m9x2m2x * pytmz, dN[13][1] =  m1mx2t1p3x * mz, dN[13][2] = -m1mx2t1p3x * py;
			dN[14][0] = m3m9x2m2x * pytpz, dN[14][1] =  p1mx2t1m3x * pz, dN[14][2] =  p1mx2t1m3x * py;
			dN[15][0] = p3m9x2m2x * pytpz, dN[15][1] =  m1mx2t1p3x * pz, dN[15][2] =  m1mx2t1p3x * py;

			const double m3m9y2m2y = -m2y92 - y2;
			const double p3m9y2m2y = m2y92 - y2;
			const double m1my2t1m3y = my2 * my3;
			const double m1my2t1p3y = my2 * py3;
			dN[16][0] = -m1my2t1m3y * mz, dN[16][1] = m3m9y2m2y * mxtmz, dN[16][2] = -m1my2t1m3y * mx;
			dN[17][0] = -m1my2t1p3y * mz, dN[17][1] = p3m9y2m2y * mxtmz, dN[17][2] = -m1my2t1p3y * mx;
			dN[18][0] =  m1my2t1m3y * mz, dN[18][1] = m3m9y2m2y * pxtmz, dN[18][2] = -m1my2t1m3y * px;
			dN[19][0] =  m1my2t1p3y * mz, dN[19][1] = p3m9y2m2y * pxtmz, dN[19][2] = -m1my2t1p3y * px;
			dN[20][0] = -m1my2t1m3y * pz, dN[20][1] = m3m9y2m2y * mxtpz, dN[20][2] =  m1my2t1m3y * mx;
			dN[21][0] = -m1my2t1p3y * pz, dN[21][1] = p3m9y2m2y * mxtpz, dN[21][2] =  m1my2t1p3y * mx;
			dN[22][0] =  m1my2t1m3y * pz, dN[22][1] = m3m9y2m2y * pxtpz, dN[22][2] =  m1my2t1m3y * px;
			dN[23][0] =  m1my2t1p3y * pz, dN[23][1] = p3m9y2m2y * pxtpz, dN[23][2] =  m1my2t1p3y * px;

			const double m3m9z2m2z = -m3z92 - z2;
			const double p3m9z2m2z = m3z92 - z2;
			const double m1mz2t1m3z = mz2 * mz3;
			const double p1mz2t1p3z = mz2 * pz3;
			dN[24][0] = -m1mz2t1m3z * my, dN[24][1] = -m1mz2t1m3z * mx, dN[24][2] = m3m9z2m2z * mxtmy;
			dN[25][0] = -p1mz2t1p3z * my, dN[25][1] = -p1mz2t1p3z * mx, dN[25][2] = p3m9z2m2z * mxtmy;
			dN[26][0] = -m1mz2t1m3z * py, dN[26][1] =  m1mz2t1m3z * mx, dN[26][2] = m3m9z2m2z * mxtpy;
			dN[27][0] = -p1mz2t1p3z * py, dN[27][1] =  p1mz2t1p3z * mx, dN[27][2] = p3m9z2m2z * mxtpy;
			dN[28][0] =  m1mz2t1m3z * my, dN[28][1] = -m1mz2t1m3z * px, dN[28][2] = m3m9z2m2z * pxtmy;
			dN[29][0] =  p1mz2t1p3z * my, dN[29][1] = -p1mz2t1p3z * px, dN[29][2] = p3m9z2m2z * pxtmy;
			dN[30][0] =  m1mz2t1m3z * py, dN[30][1] =  m1mz2t1m3z * px, dN[30][2] = m3m9z2m2z * pxtpy;
			dN[31][0] =  p1mz2t1p3z * py, dN[31][1] =  p1mz2t1p3z * px, dN[31][2] = p3m9z2m2z * pxtpy;

			const double tt = 9.0 / 64.0;
			dN[31][0] *= tt;
			dN[31][1] *= tt;
			dN[31][2] *= tt;
			dN[30][0] *= tt;
			dN[30][1] *= tt;
			dN[30][2] *= tt;
			dN[29][0] *= tt;
			dN[29][1] *= tt;
			dN[29][2] *= tt;
			dN[28][0] *= tt;
			dN[28][1] *= tt;
			dN[28][2] *= tt;
			dN[27][0] *= tt;
			dN[27][1] *= tt;
			dN[27][2] *= tt;
			dN[26][0] *= tt;
			dN[26][1] *= tt;
			dN[26][2] *= tt;
			dN[25][0] *= tt;
			dN[25][1] *= tt;
			dN[25][2] *= tt;
			dN[24][0] *= tt;
			dN[24][1] *= tt;
			dN[24][2] *= tt;
			dN[23][0] *= tt;
			dN[23][1] *= tt;
			dN[23][2] *= tt;
			dN[22][0] *= tt;
			dN[22][1] *= tt;
			dN[22][2] *= tt;
			dN[21][0] *= tt;
			dN[21][1] *= tt;
			dN[21][2] *= tt;
			dN[20][0] *= tt;
			dN[20][1] *= tt;
			dN[20][2] *= tt;
			dN[19][0] *= tt;
			dN[19][1] *= tt;
			dN[19][2] *= tt;
			dN[18][0] *= tt;
			dN[18][1] *= tt;
			dN[18][2] *= tt;
			dN[17][0] *= tt;
			dN[17][1] *= tt;
			dN[17][2] *= tt;
			dN[16][0] *= tt;
			dN[16][1] *= tt;
			dN[16][2] *= tt;
			dN[15][0] *= tt;
			dN[15][1] *= tt;
			dN[15][2] *= tt;
			dN[14][0] *= tt;
			dN[14][1] *= tt;
			dN[14][2] *= tt;
			dN[13][0] *= tt;
			dN[13][1] *= tt;
			dN[13][2] *= tt;
			dN[11][0] *= tt;
			dN[11][1] *= tt;
			dN[11][2] *= tt;
			dN[10][0] *= tt;
			dN[10][1] *= tt;
			dN[10][2] *= tt;
			dN[9][0]  *= tt;
			dN[9][1]  *= tt;
			dN[9][2]  *= tt;
			dN[8][0]  *= tt;
			dN[8][1]  *= tt;
			dN[8][2]  *= tt;
			dN[7][0]  *= tt;
			dN[7][1]  *= tt;
			dN[7][2]  *= tt;
		}

		return res;
	}

	float SDF::GetDistance(const glm::vec3& point, const float thickness) const
	{
		const float distance = Interpolate(0, point);
		if (distance == std::numeric_limits<float>::max()) {
			return distance;
		}

		return distance - thickness;
	}
}