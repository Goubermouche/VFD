#include "pch.h"
#include "SDF.h"

#include "MeshDistance.h"

namespace fe {
	SDF::SDF(const EdgeMesh& mesh, const BoundingBox& bounds, const glm::ivec3& resolution, const bool inverted)
		: m_Resolution(resolution)
	{
		MeshDistance distance(mesh);

		m_Domain.Extend(bounds.min);
		m_Domain.Extend(bounds.max);
		m_Domain.max += 0.001f * glm::sqrt(glm::dot(m_Domain.Diagonal(), m_Domain.Diagonal()));
		m_Domain.min -= 0.001f * glm::sqrt(glm::dot(m_Domain.Diagonal(), m_Domain.Diagonal()));

		m_CellSize = m_Domain.Diagonal() / (glm::vec3)m_Resolution;
		m_CellSizeInverse = 1.0f / m_CellSize;
		m_CellCount = m_Resolution.x * m_Resolution.y * m_Resolution.z;

		float factor = 1.0f;
		if (inverted) {
			factor = -1.0f;
		}

		ContinuousFunction function = [&distance, &factor](const glm::vec3& xi) {
			return factor * distance.SignedDistanceCached(xi);
		};

		AddFunction(function);
	}

	uint32_t SDF::AddFunction(const ContinuousFunction& function)
	{
		uint32_t nv = (m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 1);

		glm::ivec3 ne = {
			(m_Resolution.x + 0) * (m_Resolution.y + 1) * (m_Resolution.z + 1),
			(m_Resolution.x + 1) * (m_Resolution.y + 0) * (m_Resolution.z + 1),
			(m_Resolution.x + 1) * (m_Resolution.y + 1) * (m_Resolution.z + 0)
		};

		uint32_t nes = ne.x + ne.y + ne.z;

		uint32_t nodeCount = nv + 2 * nes;

		m_Nodes.resize(nodeCount);

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) nowait
			for (int l = 0; l < static_cast<int>(nodeCount); ++l) {
				glm::vec3 x = IndexToNodePosition(l);
				float& c = m_Nodes[l];
				c = function(x);
			}
		}

		m_Cells.resize(m_CellCount);

		for (uint32_t l = 0; l < m_CellCount; ++l)
		{
			uint32_t k = l / (m_Resolution[1] * m_Resolution[0]);
			uint32_t temp = l % (m_Resolution[1] * m_Resolution[0]);
			uint32_t j = temp / m_Resolution[0];
			uint32_t i = temp % m_Resolution[0];

			uint32_t nx = m_Resolution[0];
			uint32_t ny = m_Resolution[1];
			uint32_t nz = m_Resolution[2];

			auto& cell = m_Cells[l];
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
		m_CellMap.resize(m_CellCount);
		std::iota(m_CellMap.begin(), m_CellMap.end(), 0);
		return static_cast<uint32_t>(m_FieldCount++);
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

			result = m_Domain.min + m_CellSize * index;
		}
		else if (i < nv + 2 * ne.x)
		{
			i -= nv;
			uint32_t e_ind = i / 2;
			index.z = e_ind / ((m_Resolution.y + 1) * m_Resolution.x);
			uint32_t temp = e_ind % (uint32_t)((m_Resolution.y + 1) * m_Resolution.x);
			index.y = temp / m_Resolution.x;
			index.x = temp % (uint32_t)m_Resolution.x;

			result = m_Domain.min + m_CellSize * index;
			result.x += (1.0f + i % 2) / 3.0f * m_CellSize.x;
		}
		else if (i < nv + 2 * (ne.x + ne.y))
		{
			i -= (nv + 2 * ne.x);
			uint32_t e_ind = i / 2;
			index.x = e_ind / ((m_Resolution.z + 1) * m_Resolution.y);
			uint32_t temp = e_ind % (uint32_t)((m_Resolution.z + 1) * m_Resolution.y);
			index.z = temp / m_Resolution.y;
			index.y = temp % (uint32_t)m_Resolution.y;

			result = m_Domain.min + m_CellSize * index;
			result.y += (1.0f + i % 2) / 3.0f * m_CellSize.y;
		}
		else
		{
			i -= (nv + 2 * (ne.x + ne.y));
			uint32_t e_ind = i / 2;
			index.y = e_ind / ((m_Resolution.x + 1) * m_Resolution.z);
			uint32_t temp = e_ind % (uint32_t)((m_Resolution.x + 1) * m_Resolution.z);
			index.x = temp / m_Resolution.z;
			index.z = temp % (uint32_t)m_Resolution.z;

			result = m_Domain.min + m_CellSize * index;
			result.z += (1.0f + i % 2) / 3.0f * m_CellSize.z;
		}

		return result;
	}

	float SDF::Interpolate(const glm::vec3& point, glm::vec3* gradient) const
	{
		if (m_Domain.Contains(point) == false) {
			return std::numeric_limits<float>::max();
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

		uint32_t index = MultiToSingleIndex(multiIndex);
		auto cellIndex = m_CellMap[index];
		if (cellIndex == std::numeric_limits<uint32_t>::max()) {
			return std::numeric_limits<float>::max();
		}

		BoundingBox subDomain = CalculateSubDomain(index);
		index = cellIndex;
		glm::vec3 d = subDomain.Diagonal();
		glm::vec3 denom = (subDomain.max - subDomain.min);
		glm::vec3 c0 = 2.0f / denom;
		glm::vec3 c1 = (subDomain.max + subDomain.min) / (denom);
		glm::vec3 xi = (c0 * point - c1);

		auto const& cell = m_Cells[index];
		if (!gradient)
		{
			float phi = 0.0f;
			auto N = ShapeFunction(xi);
			for (uint32_t j = 0; j < 32; ++j)
			{
				uint32_t v = cell[j];
				float c = m_Nodes[v];
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

		for (uint32_t j = 0; j < 32; ++j)
		{
			uint32_t v = cell[j];
			float c = m_Nodes[v];

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

	glm::ivec3 SDF::SingleToMultiIndex(uint32_t index) const
	{
		uint32_t n01 = m_Resolution.x * m_Resolution.y;
		uint32_t k = index / n01;
		uint32_t temp = index % n01;
		uint32_t j = temp / m_Resolution.x;
		uint32_t i = temp % (uint32_t)m_Resolution.x;

		return { i, j ,k };
	}

	uint32_t SDF::MultiToSingleIndex(const glm::ivec3& index) const
	{
		return m_Resolution.y * m_Resolution.x * index.z + m_Resolution.x * index.y + index.x;
	}

	BoundingBox SDF::CalculateSubDomain(const glm::vec3& index) const
	{
		glm::vec3 origin = m_Domain.min + index * m_CellSize;
		BoundingBox box;
		box.min = origin;
		box.max = origin + m_CellSize;
		return box;
	}

	BoundingBox SDF::CalculateSubDomain(uint32_t index) const
	{
		return CalculateSubDomain(SingleToMultiIndex(index));
	}

	std::array<float, 32> SDF::ShapeFunction(const glm::vec3& xi, std::array<std::array<float, 3>, 32>* gradient)
	{
		auto res = std::array<float, 32>{};

		float x = xi[0];
		float y = xi[1];
		float z = xi[2];

		float x2 = x * x;
		float y2 = y * y;
		float z2 = z * z;

		float mx = 1.0f - x;
		float my = 1.0f - y;
		float mz = 1.0f - z;

		float mx2 = 1.0f - x2;
		float my2 = 1.0f - y2;
		float mz2 = 1.0f - z2;

		float px = 1.0f + x;
		float py = 1.0f + y;
		float pz = 1.0f + z;

		float mx3 = 1.0f - 3.0f * x;
		float my3 = 1.0f - 3.0f * y;
		float mz3 = 1.0f - 3.0f * z;

		float px3 = 1.0f + 3.0f * x;
		float py3 = 1.0f + 3.0f * y;
		float pz3 = 1.0f + 3.0f * z;

		float mxtmy = mx * my;
		float mxtpy = mx * py;
		float pxtmy = px * my;
		float pxtpy = px * py;

		float mxtmz = mx * mz;
		float mxtpz = mx * pz;
		float pxtmz = px * mz;
		float pxtpz = px * pz;

		float mytmz = my * mz;
		float mytpz = my * pz;
		float pytmz = py * mz;
		float pytpz = py * pz;

		// Corners
		float fac = 1.0f / 64.0f * (9.0f * (x2 + y2 + z2) - 19.0f);
		res[0] = fac * mxtmy * mz;
		res[1] = fac * pxtmy * mz;
		res[2] = fac * mxtpy * mz;
		res[3] = fac * pxtpy * mz;
		res[4] = fac * mxtmy * pz;
		res[5] = fac * pxtmy * pz;
		res[6] = fac * mxtpy * pz;
		res[7] = fac * pxtpy * pz;

		// Edges
		fac = 9.0f / 64.0f * mx2;
		float factmx3 = fac * mx3;
		float factpx3 = fac * px3;
		res[8] = factmx3 * mytmz;
		res[9] = factpx3 * mytmz;
		res[10] = factmx3 * mytpz;
		res[11] = factpx3 * mytpz;
		res[12] = factmx3 * pytmz;
		res[13] = factpx3 * pytmz;
		res[14] = factmx3 * pytpz;
		res[15] = factpx3 * pytpz;

		fac = 9.0f / 64.0f * my2;
		float factmy3 = fac * my3;
		float factpy3 = fac * py3;
		res[16] = factmy3 * mxtmz;
		res[17] = factpy3 * mxtmz;
		res[18] = factmy3 * pxtmz;
		res[19] = factpy3 * pxtmz;
		res[20] = factmy3 * mxtpz;
		res[21] = factpy3 * mxtpz;
		res[22] = factmy3 * pxtpz;
		res[23] = factpy3 * pxtpz;

		fac = 9.0f / 64.0f * mz2;
		float factmz3 = fac * mz3;
		float factpz3 = fac * pz3;
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

			float t9x3py2pzy2m19 = 9.0f * (3.0f * x2 + y2 + z2) - 19.0f;
			float t9x2p3y2pz2m19 = 9.0f * (x2 + 3.0f * y2 + z2) - 19.0f;
			float t9x2py2p3z2m19 = 9.0f * (x2 + y2 + 3.0f * z2) - 19.0f;
			float x18 = 18.0f * x;
			float y18 = 18.0f * y;
			float z18 = 18.0f * z;

			float m2x92 = 3.0f - 9.0f * x2;
			float m2y92 = 3.0f - 9.0f * y2;
			float m3z92 = 3.0f - 9.0f * z2;

			float x2 = 2.0f * x;
			float y2 = 2.0f * y;
			float z2 = 2.0f * z;

			float x18xm9t3x2py2pz2m19 = x18 - t9x3py2pzy2m19;
			float y18xp9t3x2py2pz2m19 = x18 + t9x3py2pzy2m19;
			float z18ym9tx2p3y2pz2m19 = y18 - t9x2p3y2pz2m19;
			float x18yp9tx2p3y2pz2m19 = y18 + t9x2p3y2pz2m19;
			float y18zm9tx2py2p3z2m19 = z18 - t9x2py2p3z2m19;
			float z18zp9tx2py2p3z2m19 = z18 + t9x2py2p3z2m19;

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

			float m3m9x2m2x = -m2x92 - x2;
			float p3m9x2m2x = m2x92 - x2;
			float p1mx2t1m3x = mx2 * mx3;
			float m1mx2t1p3x = mx2 * px3;
			dN[8][0]  = m3m9x2m2x * mytmz, dN[8][1]  = -p1mx2t1m3x * mz, dN[8][2]  = -p1mx2t1m3x * my;
			dN[9][0]  = p3m9x2m2x * mytmz, dN[9][1]  = -m1mx2t1p3x * mz, dN[9][2]  = -m1mx2t1p3x * my;
			dN[10][0] = m3m9x2m2x * mytpz, dN[10][1] = -p1mx2t1m3x * pz, dN[10][2] =  p1mx2t1m3x * my;
			dN[11][0] = p3m9x2m2x * mytpz, dN[11][1] = -m1mx2t1p3x * pz, dN[11][2] =  m1mx2t1p3x * my;
			dN[12][0] = m3m9x2m2x * pytmz, dN[12][1] =  p1mx2t1m3x * mz, dN[12][2] = -p1mx2t1m3x * py;
			dN[13][0] = p3m9x2m2x * pytmz, dN[13][1] =  m1mx2t1p3x * mz, dN[13][2] = -m1mx2t1p3x * py;
			dN[14][0] = m3m9x2m2x * pytpz, dN[14][1] =  p1mx2t1m3x * pz, dN[14][2] =  p1mx2t1m3x * py;
			dN[15][0] = p3m9x2m2x * pytpz, dN[15][1] =  m1mx2t1p3x * pz, dN[15][2] =  m1mx2t1p3x * py;

			float m3m9y2m2y = -m2y92 - y2;
			float p3m9y2m2y = m2y92 - y2;
			float m1my2t1m3y = my2 * my3;
			float m1my2t1p3y = my2 * py3;
			dN[16][0] = -m1my2t1m3y * mz, dN[16][1] = m3m9y2m2y * mxtmz, dN[16][2] = -m1my2t1m3y * mx;
			dN[17][0] = -m1my2t1p3y * mz, dN[17][1] = p3m9y2m2y * mxtmz, dN[17][2] = -m1my2t1p3y * mx;
			dN[18][0] =  m1my2t1m3y * mz, dN[18][1] = m3m9y2m2y * pxtmz, dN[18][2] = -m1my2t1m3y * px;
			dN[19][0] =  m1my2t1p3y * mz, dN[19][1] = p3m9y2m2y * pxtmz, dN[19][2] = -m1my2t1p3y * px;
			dN[20][0] = -m1my2t1m3y * pz, dN[20][1] = m3m9y2m2y * mxtpz, dN[20][2] =  m1my2t1m3y * mx;
			dN[21][0] = -m1my2t1p3y * pz, dN[21][1] = p3m9y2m2y * mxtpz, dN[21][2] =  m1my2t1p3y * mx;
			dN[22][0] =  m1my2t1m3y * pz, dN[22][1] = m3m9y2m2y * pxtpz, dN[22][2] =  m1my2t1m3y * px;
			dN[23][0] =  m1my2t1p3y * pz, dN[23][1] = p3m9y2m2y * pxtpz, dN[23][2] =  m1my2t1p3y * px;

			float m3m9z2m2z = -m3z92 - z2;
			float p3m9z2m2z = m3z92 - z2;
			float m1mz2t1m3z = mz2 * mz3;
			float p1mz2t1p3z = mz2 * pz3;
			dN[24][0] = -m1mz2t1m3z * my, dN[24][1] = -m1mz2t1m3z * mx, dN[24][2] = m3m9z2m2z * mxtmy;
			dN[25][0] = -p1mz2t1p3z * my, dN[25][1] = -p1mz2t1p3z * mx, dN[25][2] = p3m9z2m2z * mxtmy;
			dN[26][0] = -m1mz2t1m3z * py, dN[26][1] =  m1mz2t1m3z * mx, dN[26][2] = m3m9z2m2z * mxtpy;
			dN[27][0] = -p1mz2t1p3z * py, dN[27][1] =  p1mz2t1p3z * mx, dN[27][2] = p3m9z2m2z * mxtpy;
			dN[28][0] =  m1mz2t1m3z * my, dN[28][1] = -m1mz2t1m3z * px, dN[28][2] = m3m9z2m2z * pxtmy;
			dN[29][0] =  p1mz2t1p3z * my, dN[29][1] = -p1mz2t1p3z * px, dN[29][2] = p3m9z2m2z * pxtmy;
			dN[30][0] =  m1mz2t1m3z * py, dN[30][1] =  m1mz2t1m3z * px, dN[30][2] = m3m9z2m2z * pxtpy;
			dN[31][0] =  p1mz2t1p3z * py, dN[31][1] =  p1mz2t1p3z * px, dN[31][2] = p3m9z2m2z * pxtpy;

			float tt = 9.0f / 64.0f;
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

	float SDF::GetDistance(const glm::vec3& point, const float thickness)
	{
		const float distance = Interpolate(point);
		if (distance == std::numeric_limits<float>::max()) {
			return distance;
		}

		return distance - thickness;
	}
}