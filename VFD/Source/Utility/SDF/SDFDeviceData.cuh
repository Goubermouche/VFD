#ifndef DENSITY_MAP_DEVICE_DATA_CUH
#define DENSITY_MAP_DEVICE_DATA_CUH

#include "Core/Structures/BoundingBox.h"

namespace vfd {
	struct SDFDeviceData
	{
		__host__ __device__ unsigned int MultiToSingleIndex(const glm::uvec3& index) const
		{
			return m_Resolution.y * m_Resolution.x * index.z + m_Resolution.x * index.y + index.x;
		}

		__host__ __device__ glm::uvec3 SingleToMultiIndex(unsigned int index) const
		{
			const unsigned int n01 = m_Resolution.x * m_Resolution.y;
			unsigned int k = index / n01;
			const unsigned int temp = index % n01;
			float j = temp / m_Resolution.x;
			float i = temp % m_Resolution.x;

			return { i, j, k };
		}

		__host__ __device__ BoundingBox<glm::vec3> CalculateSubDomain(const glm::uvec3& index) const
		{
			const glm::vec3 origin = m_Domain.min + (static_cast<glm::vec3>(index) * m_CellSize);
			return { origin, origin + m_CellSize };
		}

		__host__ __device__ BoundingBox<glm::vec3> CalculateSubDomain(unsigned int index) const
		{
			return CalculateSubDomain(SingleToMultiIndex(index));
		}

		__host__ __device__ static void ShapeFunction(
			float(&res)[32],
			const glm::vec3& xi,
			glm::vec3(&gradient)[32]
		)
		{
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

				gradient[0][0] = _18xm9t3x2py2pz2m19 * _1myt1mz;
				gradient[0][1] = _1mxt1mz * _18ym9tx2p3y2pz2m19;
				gradient[0][2] = _1mxt1my * _18zm9tx2py2p3z2m19;
				gradient[1][0] = _18xp9t3x2py2pz2m19 * _1myt1mz;
				gradient[1][1] = _1pxt1mz * _18ym9tx2p3y2pz2m19;
				gradient[1][2] = _1pxt1my * _18zm9tx2py2p3z2m19;
				gradient[2][0] = _18xm9t3x2py2pz2m19 * _1pyt1mz;
				gradient[2][1] = _1mxt1mz * _18yp9tx2p3y2pz2m19;
				gradient[2][2] = _1mxt1py * _18zm9tx2py2p3z2m19;
				gradient[3][0] = _18xp9t3x2py2pz2m19 * _1pyt1mz;
				gradient[3][1] = _1pxt1mz * _18yp9tx2p3y2pz2m19;
				gradient[3][2] = _1pxt1py * _18zm9tx2py2p3z2m19;
				gradient[4][0] = _18xm9t3x2py2pz2m19 * _1myt1pz;
				gradient[4][1] = _1mxt1pz * _18ym9tx2p3y2pz2m19;
				gradient[4][2] = _1mxt1my * _18zp9tx2py2p3z2m19;
				gradient[5][0] = _18xp9t3x2py2pz2m19 * _1myt1pz;
				gradient[5][1] = _1pxt1pz * _18ym9tx2p3y2pz2m19;
				gradient[5][2] = _1pxt1my * _18zp9tx2py2p3z2m19;
				gradient[6][0] = _18xm9t3x2py2pz2m19 * _1pyt1pz;
				gradient[6][1] = _1mxt1pz * _18yp9tx2p3y2pz2m19;
				gradient[6][2] = _1mxt1py * _18zp9tx2py2p3z2m19;
				gradient[7][0] = _18xp9t3x2py2pz2m19 * _1pyt1pz;
				gradient[7][1] = _1pxt1pz * _18yp9tx2p3y2pz2m19;
				gradient[7][2] = _1pxt1py * _18zp9tx2py2p3z2m19;

				gradient[0][0] /= 64.0f;
				gradient[0][1] /= 64.0f;
				gradient[0][2] /= 64.0f;
				gradient[1][0] /= 64.0f;
				gradient[1][1] /= 64.0f;
				gradient[1][2] /= 64.0f;
				gradient[2][0] /= 64.0f;
				gradient[2][1] /= 64.0f;
				gradient[2][2] /= 64.0f;
				gradient[3][0] /= 64.0f;
				gradient[3][1] /= 64.0f;
				gradient[3][2] /= 64.0f;
				gradient[4][0] /= 64.0f;
				gradient[4][1] /= 64.0f;
				gradient[4][2] /= 64.0f;
				gradient[5][0] /= 64.0f;
				gradient[5][1] /= 64.0f;
				gradient[5][2] /= 64.0f;
				gradient[6][0] /= 64.0f;
				gradient[6][1] /= 64.0f;
				gradient[6][2] /= 64.0f;
				gradient[7][0] /= 64.0f;
				gradient[7][1] /= 64.0f;
				gradient[7][2] /= 64.0f;

				const float _m3m9x2m2x = -_3m9x2 - _2x;
				const float _p3m9x2m2x = _3m9x2 - _2x;
				const float _1mx2t1m3x = _1mx2 * _1m3x;
				const float _1mx2t1p3x = _1mx2 * _1p3x;

				gradient[8][0] = _m3m9x2m2x * _1myt1mz;
				gradient[8][1] = -_1mx2t1m3x * _1mz;
				gradient[8][2] = -_1mx2t1m3x * _1my;
				gradient[9][0] = _p3m9x2m2x * _1myt1mz;
				gradient[9][1] = -_1mx2t1p3x * _1mz;
				gradient[9][2] = -_1mx2t1p3x * _1my;
				gradient[10][0] = _m3m9x2m2x * _1myt1pz;
				gradient[10][1] = -_1mx2t1m3x * _1pz;
				gradient[10][2] = _1mx2t1m3x * _1my;
				gradient[11][0] = _p3m9x2m2x * _1myt1pz;
				gradient[11][1] = -_1mx2t1p3x * _1pz;
				gradient[11][2] = _1mx2t1p3x * _1my;
				gradient[12][0] = _m3m9x2m2x * _1pyt1mz;
				gradient[12][1] = _1mx2t1m3x * _1mz;
				gradient[12][2] = -_1mx2t1m3x * _1py;
				gradient[13][0] = _p3m9x2m2x * _1pyt1mz;
				gradient[13][1] = _1mx2t1p3x * _1mz;
				gradient[13][2] = -_1mx2t1p3x * _1py;
				gradient[14][0] = _m3m9x2m2x * _1pyt1pz;
				gradient[14][1] = _1mx2t1m3x * _1pz;
				gradient[14][2] = _1mx2t1m3x * _1py;
				gradient[15][0] = _p3m9x2m2x * _1pyt1pz;
				gradient[15][1] = _1mx2t1p3x * _1pz;
				gradient[15][2] = _1mx2t1p3x * _1py;

				const float _m3m9y2m2y = -_3m9y2 - _2y;
				const float _p3m9y2m2y = _3m9y2 - _2y;
				const float _1my2t1m3y = _1my2 * _1m3y;
				const float _1my2t1p3y = _1my2 * _1p3y;

				gradient[16][0] = -_1my2t1m3y * _1mz;
				gradient[16][1] = _m3m9y2m2y * _1mxt1mz;
				gradient[16][2] = -_1my2t1m3y * _1mx;
				gradient[17][0] = -_1my2t1p3y * _1mz;
				gradient[17][1] = _p3m9y2m2y * _1mxt1mz;
				gradient[17][2] = -_1my2t1p3y * _1mx;
				gradient[18][0] = _1my2t1m3y * _1mz;
				gradient[18][1] = _m3m9y2m2y * _1pxt1mz;
				gradient[18][2] = -_1my2t1m3y * _1px;
				gradient[19][0] = _1my2t1p3y * _1mz;
				gradient[19][1] = _p3m9y2m2y * _1pxt1mz;
				gradient[19][2] = -_1my2t1p3y * _1px;
				gradient[20][0] = -_1my2t1m3y * _1pz;
				gradient[20][1] = _m3m9y2m2y * _1mxt1pz;
				gradient[20][2] = _1my2t1m3y * _1mx;
				gradient[21][0] = -_1my2t1p3y * _1pz;
				gradient[21][1] = _p3m9y2m2y * _1mxt1pz;
				gradient[21][2] = _1my2t1p3y * _1mx;
				gradient[22][0] = _1my2t1m3y * _1pz;
				gradient[22][1] = _m3m9y2m2y * _1pxt1pz;
				gradient[22][2] = _1my2t1m3y * _1px;
				gradient[23][0] = _1my2t1p3y * _1pz;
				gradient[23][1] = _p3m9y2m2y * _1pxt1pz;
				gradient[23][2] = _1my2t1p3y * _1px;

				const float _m3m9z2m2z = -_3m9z2 - _2z;
				const float _p3m9z2m2z = _3m9z2 - _2z;
				const float _1mz2t1m3z = _1mz2 * _1m3z;
				const float _1mz2t1p3z = _1mz2 * _1p3z;

				gradient[24][0] = -_1mz2t1m3z * _1my;
				gradient[24][1] = -_1mz2t1m3z * _1mx;
				gradient[24][2] = _m3m9z2m2z * _1mxt1my;
				gradient[25][0] = -_1mz2t1p3z * _1my;
				gradient[25][1] = -_1mz2t1p3z * _1mx;
				gradient[25][2] = _p3m9z2m2z * _1mxt1my;
				gradient[26][0] = -_1mz2t1m3z * _1py;
				gradient[26][1] = _1mz2t1m3z * _1mx;
				gradient[26][2] = _m3m9z2m2z * _1mxt1py;
				gradient[27][0] = -_1mz2t1p3z * _1py;
				gradient[27][1] = _1mz2t1p3z * _1mx;
				gradient[27][2] = _p3m9z2m2z * _1mxt1py;
				gradient[28][0] = _1mz2t1m3z * _1my;
				gradient[28][1] = -_1mz2t1m3z * _1px;
				gradient[28][2] = _m3m9z2m2z * _1pxt1my;
				gradient[29][0] = _1mz2t1p3z * _1my;
				gradient[29][1] = -_1mz2t1p3z * _1px;
				gradient[29][2] = _p3m9z2m2z * _1pxt1my;
				gradient[30][0] = _1mz2t1m3z * _1py;
				gradient[30][1] = _1mz2t1m3z * _1px;
				gradient[30][2] = _m3m9z2m2z * _1pxt1py;
				gradient[31][0] = _1mz2t1p3z * _1py;
				gradient[31][1] = _1mz2t1p3z * _1px;
				gradient[31][2] = _p3m9z2m2z * _1pxt1py;

				constexpr float rfe = 9.0f / 64.0f;
				gradient[31][0] *= rfe;
				gradient[31][1] *= rfe;
				gradient[31][2] *= rfe;
				gradient[30][0] *= rfe;
				gradient[30][1] *= rfe;
				gradient[30][2] *= rfe;
				gradient[29][0] *= rfe;
				gradient[29][1] *= rfe;
				gradient[29][2] *= rfe;
				gradient[28][0] *= rfe;
				gradient[28][1] *= rfe;
				gradient[28][2] *= rfe;
				gradient[27][0] *= rfe;
				gradient[27][1] *= rfe;
				gradient[27][2] *= rfe;
				gradient[26][0] *= rfe;
				gradient[26][1] *= rfe;
				gradient[26][2] *= rfe;
				gradient[25][0] *= rfe;
				gradient[25][1] *= rfe;
				gradient[25][2] *= rfe;
				gradient[24][0] *= rfe;
				gradient[24][1] *= rfe;
				gradient[24][2] *= rfe;
				gradient[23][0] *= rfe;
				gradient[23][1] *= rfe;
				gradient[23][2] *= rfe;
				gradient[22][0] *= rfe;
				gradient[22][1] *= rfe;
				gradient[22][2] *= rfe;
				gradient[21][0] *= rfe;
				gradient[21][1] *= rfe;
				gradient[21][2] *= rfe;
				gradient[20][0] *= rfe;
				gradient[20][1] *= rfe;
				gradient[20][2] *= rfe;
				gradient[19][0] *= rfe;
				gradient[19][1] *= rfe;
				gradient[19][2] *= rfe;
				gradient[18][0] *= rfe;
				gradient[18][1] *= rfe;
				gradient[18][2] *= rfe;
				gradient[17][0] *= rfe;
				gradient[17][1] *= rfe;
				gradient[17][2] *= rfe;
				gradient[16][0] *= rfe;
				gradient[16][1] *= rfe;
				gradient[16][2] *= rfe;
				gradient[15][0] *= rfe;
				gradient[15][1] *= rfe;
				gradient[15][2] *= rfe;
				gradient[14][0] *= rfe;
				gradient[14][1] *= rfe;
				gradient[14][2] *= rfe;
				gradient[13][0] *= rfe;
				gradient[13][1] *= rfe;
				gradient[13][2] *= rfe;
				gradient[12][0] *= rfe;
				gradient[12][1] *= rfe;
				gradient[12][2] *= rfe;
				gradient[11][0] *= rfe;
				gradient[11][1] *= rfe;
				gradient[11][2] *= rfe;
				gradient[10][0] *= rfe;
				gradient[10][1] *= rfe;
				gradient[10][2] *= rfe;
				gradient[9 ][0] *= rfe;
				gradient[9 ][1] *= rfe;
				gradient[9 ][2] *= rfe;
				gradient[8 ][0] *= rfe;
				gradient[8 ][1] *= rfe;
				gradient[8 ][2] *= rfe;
			}
		}

		__host__ __device__ bool DetermineShapeFunction(
			unsigned int fieldID,
			const glm::vec3& x,
			unsigned int(&cell)[32],
			glm::vec3& c0,
			float(&N)[32],
			glm::vec3(&dN)[32]
		)
		{
			if (m_Domain.Contains(x) == false) {
				return false;
			}

			glm::uvec3 mi = m_CellSizeInverse * (x - m_Domain.min);

			if (mi[0] >= m_Resolution[0]) {
				mi[0] = m_Resolution[0] - 1u;
			}

			if (mi[1] >= m_Resolution[1]) {
				mi[1] = m_Resolution[1] - 1u;
			}

			if (mi[2] >= m_Resolution[2]) {
				mi[2] = m_Resolution[2] - 1u;
			}

			const unsigned int i = MultiToSingleIndex(mi);
			const unsigned int j = GetCellMap(fieldID, i);

			if (j == UINT_MAX) {
				return false;
			}

			const BoundingBox<glm::vec3> sd = CalculateSubDomain(i);
			const glm::vec3 denominator = sd.Diagonal();
			c0 = 2.0f / denominator;
			const glm::vec3 c1 = (sd.max + sd.min) / denominator;
			const glm::vec3 xi = c0 * x - c1;

			for (unsigned int idx = 0; idx < 32u; idx++)
			{
				cell[idx] = GetCell(fieldID, j, idx);
			}

			ShapeFunction(N, xi, dN);
			return true;
		}

		__host__ __device__ float Interpolate(
			unsigned int fieldID,
			const unsigned int(&cell)[32],
			const float(&N)[32]
		)
		{
			float phi = 0.0;

			for (unsigned int j = 0u; j < 32u; ++j)
			{
				const unsigned int v = cell[j];
				const float c = GetNode(fieldID, v);

				if (c == DBL_MAX)
				{
					return DBL_MAX;
				}

				phi += c * N[j];
			}

			return phi;
		}

		__host__ __device__ float Interpolate(
			unsigned int fieldID,
			const unsigned int(&cell)[32],
			const glm::vec3& c0,
			const float(&N)[32],
			glm::vec3& gradient,
			const glm::vec3(&dN)[32]
		)
		{
			float phi = 0.0f;
			gradient = { 0.0f, 0.0f, 0.0f };

			for (unsigned int j = 0u; j < 32u; ++j)
			{
				const unsigned int v = cell[j];
				const float c = GetNode(fieldID, v);

				if (c == DBL_MAX)
				{
					gradient = { 0.0f, 0.0f, 0.0f };
					return DBL_MAX;
				}

				phi += c * N[j];
				gradient += c * dN[j];
			}
			gradient *= c0;

			return phi;
		}

		__host__ __device__ __forceinline__ float GetNode(unsigned int i, unsigned int j) const
		{
			return m_Nodes[i * m_NodeCount + j];
		}

		__host__ __device__ __forceinline__ float& GetNode(unsigned int i, unsigned int j)
		{
			return m_Nodes[i * m_NodeCount + j];
		}

		__host__ __device__ __forceinline__ unsigned  int GetCellMap(unsigned int i, unsigned int j) const
		{
			return m_CellMap[i * m_CellMapCount + j];
		}

		__host__ __device__ __forceinline__ unsigned int& GetCellMap(unsigned int i, unsigned int j)
		{
			return m_CellMap[i * m_CellMapCount + j];
		}

		__host__ __device__ __forceinline__ unsigned  int GetCell(
			unsigned int i,
			unsigned int j,
			unsigned int k
		) const
		{
			return m_Cells[i * m_CellCount * 32u + (j * 32u + k)];
		}

		__host__ __device__ __forceinline__ unsigned int& GetCell(
			unsigned int i,
			unsigned int j,
			unsigned int k
		)
		{
			return m_Cells[i * m_CellCount * 32u + (j * 32u + k)];
		}

		__host__ __device__ unsigned int GetNodeElementCount() const
		{
			return m_NodeCount;
		}

		__host__ __device__ float* GetNodes() const
		{
			return m_Nodes;
		}

		__host__ __device__ unsigned int GetCellMapElementCount() const
		{
			return m_CellMapCount;
		}

		__host__ __device__ unsigned int* GetCellMap() const
		{
			return m_CellMap;
		}

		__host__ __device__ unsigned int GetCellElementCount() const
		{
			return m_CellCount;
		}

		__host__ __device__ unsigned int* GetCells() const
		{
			return m_Cells;
		}

		__host__ __device__ unsigned int GetFieldCount() const
		{
			return m_FieldCount;
		}

	private:
		friend struct SDF;

		float* m_Nodes;
		unsigned int* m_Cells;
		unsigned int* m_CellMap;

		BoundingBox<glm::vec3> m_Domain;

		glm::uvec3 m_Resolution;
		glm::vec3 m_CellSize;
		glm::vec3 m_CellSizeInverse;

		unsigned int m_FieldCount;
		unsigned int m_NodeCount;
		unsigned int m_CellCount;
		unsigned int m_CellMapCount;
	};
}

#endif // !DENSITY_MAP_DEVICE_DATA_CUH