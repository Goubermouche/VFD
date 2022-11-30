#include "pch.h"
#include "GPUSDF.h"

//#include "stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

namespace vfd {
	GPUSDF::GPUSDF(Ref<TriangleMesh>& mesh) {
		const std::vector<glm::vec3>& vertices = mesh->GetVertices();
		const std::vector<glm::uvec3>& triangles = mesh->GetTriangles();

		m_Domain = BoundingBox(vertices);
		m_Domain.min -= padding * m_CellSize;
		m_Domain.max += padding * m_CellSize;

		m_CellSizeInverse = 1.0f / m_CellSize;
		m_CellCount = glm::compMul(m_Resolution);
		m_Resolution = glm::ceil((m_Domain.max - m_Domain.min) / m_CellSize);

		m_PHI.Resize(m_Resolution, (m_Resolution.x + m_Resolution.y + m_Resolution.z) * m_CellSize);

		Array3D<int> closestTriangles(m_Resolution, -1);
		Array3D<int> intersectionCounts(m_Resolution, 0);

		glm::vec3 ijkmin;
		glm::vec3 ijkmax;

		for (unsigned int t = 0u; t < triangles.size(); ++t) {
			const glm::uvec3 tri = triangles[t];

			// High precision grid coordinates
			const double fip = (static_cast<double>(vertices[tri.x][0]) - m_Domain.min[0]) / m_CellSize;
			const double fjp = (static_cast<double>(vertices[tri.x][1]) - m_Domain.min[1]) / m_CellSize;
			const double fkp = (static_cast<double>(vertices[tri.x][2]) - m_Domain.min[2]) / m_CellSize;
			const double fiq = (static_cast<double>(vertices[tri.y][0]) - m_Domain.min[0]) / m_CellSize;
			const double fjq = (static_cast<double>(vertices[tri.y][1]) - m_Domain.min[1]) / m_CellSize;
			const double fkq = (static_cast<double>(vertices[tri.y][2]) - m_Domain.min[2]) / m_CellSize;
			const double fir = (static_cast<double>(vertices[tri.z][0]) - m_Domain.min[0]) / m_CellSize; 
			const double fjr = (static_cast<double>(vertices[tri.z][1]) - m_Domain.min[1]) / m_CellSize; 
			const double fkr = (static_cast<double>(vertices[tri.z][2]) - m_Domain.min[2]) / m_CellSize;

			// Distances nearby
			int i0 = std::clamp(static_cast<int>(std::min(std::min(fip, fiq), fir))    , 0, m_Resolution.x - 1); 
			int i1 = std::clamp(static_cast<int>(std::max(std::max(fip, fiq), fir)) + 1, 0, m_Resolution.x - 1);
			int j0 = std::clamp(static_cast<int>(std::min(std::min(fjp, fjq), fjr))    , 0, m_Resolution.y - 1); 
			int j1 = std::clamp(static_cast<int>(std::max(std::max(fjp, fjq), fjr)) + 1, 0, m_Resolution.y - 1);
			int k0 = std::clamp(static_cast<int>(std::min(std::min(fkp, fkq), fkr))    , 0, m_Resolution.z - 1); 
			int k1 = std::clamp(static_cast<int>(std::max(std::max(fkp, fkq), fkr)) + 1, 0, m_Resolution.z - 1);

			for (int k = k0; k <= k1; ++k) {
				for (int j = j0; j <= j1; ++j) {
					for (int i = i0; i <= i1; ++i) {
						const glm::vec3 gx(i * m_CellSize + m_Domain.min[0], j * m_CellSize + m_Domain.min[1], k * m_CellSize + m_Domain.min[2]);
						const float d = PointToTriangleDistance(gx, vertices[tri.x], vertices[tri.y], vertices[tri.z]);

						if (d < m_PHI(i, j, k)) {
							m_PHI(i, j, k) = d;
							closestTriangles(i, j, k) = t;
						}
					}
				}
			}
				
			// Intersection counts
			j0 = std::clamp(static_cast<int>(std::ceil (std::min(std::min(fjp, fjq), fjr))), 0, m_Resolution.y - 1);
			j1 = std::clamp(static_cast<int>(std::floor(std::max(std::max(fjp, fjq), fjr))), 0, m_Resolution.y - 1);
			k0 = std::clamp(static_cast<int>(std::ceil (std::min(std::min(fkp, fkq), fkr))), 0, m_Resolution.z - 1);
			k1 = std::clamp(static_cast<int>(std::floor(std::max(std::max(fkp, fkq), fkr))), 0, m_Resolution.z - 1);

			for (int k = k0; k <= k1; ++k) {
				for (int j = j0; j <= j1; ++j) {
					double a;
					double b;
					double c;

					if (PointInTriangle2D(j, k, fjp, fkp, fjq, fkq, fjr, fkr, a, b, c)) {
						const double fi = a * fip + b * fiq + c * fir;
						const int interval = static_cast<int>(std::ceil(fi));

						if (interval < 0) {
							++intersectionCounts(0, j, k);
						}
						else if (interval < m_Resolution.x) {
							++intersectionCounts(interval, j, k);
						}
					}
				}
			}
		}

		// Fast sweep
		for (uint8_t pass = 0u; pass < 2u; ++pass) {
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, +1, +1, +1);
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, -1, -1, -1);
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, +1, +1, -1);
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, -1, -1, +1);
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, +1, -1, +1);
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, -1, +1, -1);
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, +1, -1, -1);
			Sweep(triangles, vertices, m_PHI, closestTriangles, m_Domain.min, m_CellSize, -1, +1, +1);
		}

		// Compute signs from intersections
		for (int k = 0; k < m_Resolution.z; ++k) {
			for (int j = 0; j < m_Resolution.y; ++j) {
				int intersectionCount = 0;

				for (int i = 0; i < m_Resolution.x; ++i) {
					intersectionCount += intersectionCounts(i, j, k);

					if (intersectionCount % 2 == 1) {
						m_PHI(i, j, k) = -m_PHI(i, j, k);
					}
				}
			}
		}

		closestTriangles.Free();
		intersectionCounts.Free();
	}

	float GPUSDF::GetDistance(const glm::vec3& point)
	{
		if (m_Domain.Contains(point) == false) {
			return std::numeric_limits<float>::max();
		}

		glm::uvec3 index = static_cast<glm::uvec3>((point - m_Domain.min) * m_CellSizeInverse);

		if (index.x >= m_Resolution.x) {
			index.x = m_Resolution.x - 1;
		}
		if (index.y >= m_Resolution.y) {
			index.y = m_Resolution.y - 1;
		}
		if (index.z >= m_Resolution.z) {
			index.z = m_Resolution.z - 1;
		}

		return m_PHI(index.x, index.y, index.z);
	}

	float GPUSDF::GetDistanceTrilinear(const glm::vec3& point)
	{
		if (m_Domain.Contains(point) == false) {
			return std::numeric_limits<float>::max();
		}

		glm::vec3 pointGridSpace = (point - m_Domain.min) * m_CellSizeInverse;
		glm::uvec3 index = static_cast<glm::uvec3>(pointGridSpace);

		if (index.x >= m_Resolution.x) {
			index.x = m_Resolution.x - 1;
		}
		if (index.y >= m_Resolution.y) {
			index.y = m_Resolution.y - 1;
		}
		if (index.z >= m_Resolution.z) {
			index.z = m_Resolution.z - 1;
		}

		glm::vec3 pointCellSpace = pointGridSpace - static_cast<glm::vec3>(index);

		float c000 = 0.0f;
		float c100 = 0.0f;
		float c010 = 0.0f;
		float c110 = 0.0f;
		float c001 = 0.0f;
		float c101 = 0.0f;
		float c011 = 0.0f;
		float c111 = 0.0f;

		if (m_PHI.IsIndexInRange(index.x, index.y, index.z)) {
			c000 = m_PHI(index.x, index.y, index.z);
		}

		if (m_PHI.IsIndexInRange(index.x + 1, index.y, index.z)) {
			c100 = m_PHI(index.x + 1, index.y, index.z);
		}

		if (m_PHI.IsIndexInRange(index.x, index.y + 1, index.z)) {
			c010 = m_PHI(index.x, index.y + 1, index.z);
		}

		if (m_PHI.IsIndexInRange(index.x + 1, index.y + 1, index.z)) {
			c110 = m_PHI(index.x + 1, index.y + 1, index.z);
		}

		if (m_PHI.IsIndexInRange(index.x, index.y, index.z + 1)) {
			c001 = m_PHI(index.x, index.y, index.z + 1);
		}

		if (m_PHI.IsIndexInRange(index.x + 1, index.y, index.z + 1)) {
			c101 = m_PHI(index.x + 1, index.y, index.z + 1);
		}

		if (m_PHI.IsIndexInRange(index.x, index.y + 1, index.z + 1)) {
			c011 = m_PHI(index.x, index.y + 1, index.z + 1);
		}

		if (m_PHI.IsIndexInRange(index.x + 1, index.y + 1, index.z + 1)) {
			c111 = m_PHI(index.x + 1, index.y + 1, index.z + 1);
		}

		return (1.0f - pointCellSpace.x) * (1.0f - pointCellSpace.y) * (1.0f - pointCellSpace.z) * c000 +
		        pointCellSpace.x * (1.0f - pointCellSpace.y) * (1.0f - pointCellSpace.z) * c100 +
		        (1.0f - pointCellSpace.x) * pointCellSpace.y * (1.0f - pointCellSpace.z) * c010 +
		        pointCellSpace.x * pointCellSpace.y * (1.0f - pointCellSpace.z) * c110 +
		        (1.0f - pointCellSpace.x) * (1.0f - pointCellSpace.y) * pointCellSpace.z * c001 +
		        pointCellSpace.x * (1.0f - pointCellSpace.y) * pointCellSpace.z * c101 +
		        (1.0f - pointCellSpace.x) * pointCellSpace.y * pointCellSpace.z * c011 +
		        pointCellSpace.x * pointCellSpace.y * pointCellSpace.z * c111;
	}

	// TODO: use glm matrices instead of arrays 
	float GPUSDF::GetDistanceTricubic(const glm::vec3& point)
	{
		if (m_Domain.Contains(point) == false) {
			return std::numeric_limits<float>::max();
		}

		glm::vec3 pointGridSpace = (point - m_Domain.min) * m_CellSizeInverse;
		glm::uvec3 index = static_cast<glm::uvec3>(pointGridSpace);

		if (index.x >= m_Resolution.x) {
			index.x = m_Resolution.x - 1;
		}
		if (index.y >= m_Resolution.y) {
			index.y = m_Resolution.y - 1;
		}
		if (index.z >= m_Resolution.z) {
			index.z = m_Resolution.z - 1;
		}

		constexpr glm::mat4x4 weights(
			1.0f / 6.0f, -3.0f / 6.0f,  3.0f / 6.0f, -1.0f / 6.0f,
			4.0f / 6.0f,  0.0f / 6.0f, -6.0f / 6.0f,  3.0f / 6.0f,
			1.0f / 6.0f,  3.0f / 6.0f,  3.0f / 6.0f, -3.0f / 6.0f,
			0.0f / 6.0f,  0.0f / 6.0f,  0.0f / 6.0f,  1.0f / 6.0f
		);

		glm::vec4 U;
		glm::vec4 V;
		glm::vec4 W;

		U.x = 1.0f;
		U.y = pointGridSpace.x - index.x;
		U.z = U.y * U.y;
		U.w = U.y * U.z;

		V.x = 1.0f;
		V.y = pointGridSpace.y - index.y;
		V.z = V.y * V.y;
		V.w = V.y * V.z;

		W.x = 1.0f;
		W.y = pointGridSpace.z - index.z;
		W.z = W.y * W.y;
		W.w = W.y * W.z;

		glm::vec4 P;
		glm::vec4 Q;
		glm::vec4 R;

		for (int32_t row = 0; row < 4; ++row)
		{
			P[row] = 0.0f;
			Q[row] = 0.0f;
			R[row] = 0.0f;

			for (int32_t col = 0; col < 4; ++col)
			{
				P[row] += weights[row][col] * U[col];
				Q[row] += weights[row][col] * V[col];
				R[row] += weights[row][col] * W[col];
			}
		}

		index--;
		float result = 0.0f;

		for (int32_t slice = 0; slice < 4; ++slice)
		{
			int32_t zClamp = index.z + slice;

			if (zClamp < 0)
			{
				zClamp = 0;
			}
			else if (zClamp > m_Resolution.z - 1)
			{
				zClamp = m_Resolution.z - 1;
			}

			for (int32_t row = 0; row < 4; ++row)
			{
				int32_t yClamp = index.y + row;

				if (yClamp < 0)
				{
					yClamp = 0;
				}
				else if (yClamp > m_Resolution.y - 1)
				{
					yClamp = m_Resolution.y - 1;
				}

				for (int32_t col = 0; col < 4; ++col)
				{
					int32_t xClamp = index.x + col;

					if (xClamp < 0)
					{
						xClamp = 0;
					}
					else if (xClamp > m_Resolution.x - 1)
					{
						xClamp = m_Resolution.x - 1;
					}

					result += P[col] * Q[row] * R[slice] * m_PHI[xClamp + m_Resolution.x * (yClamp + m_Resolution.y * zClamp)];
				}
			}
		}

		return result;
	}

	const BoundingBox<glm::vec3>& GPUSDF::GetDomain()
	{
		return m_Domain;
	}

	float GPUSDF::PointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3)
	{
		const glm::vec3 x13(x1 - x3);
		const glm::vec3 x23(x2 - x3);
		const glm::vec3 x03(x0 - x3);

		const float m13 = glm::length(x13);
		const float m23 = glm::length(x23);
		const float d = glm::dot(x13, x23);
		const float invdet = 1.f / std::max(m13 * m23 - d * d, 1e-30f);
		const float a = dot(x13, x03);
		const float b = dot(x23, x03);

		const float w23 = invdet * (m23 * a - d * b);
		const float w31 = invdet * (m13 * b - d * a);
		const float w12 = 1.0f - w23 - w31;

		// Inside
		if (w23 >= 0.0f && w31 >= 0.0f && w12 >= 0.0f) {
			return glm::distance(x0, w23 * x1 + w31 * x2 + w12 * x3);
		}
		// Outside?
		else {
			if (w23 > 0.0f) {
				return std::min(PointToSegmentDistance(x0, x1, x2), PointToSegmentDistance(x0, x1, x3));
			}
			else if (w31 > 0.0f) {
				return std::min(PointToSegmentDistance(x0, x1, x2), PointToSegmentDistance(x0, x2, x3));
			}
			else {
				return std::min(PointToSegmentDistance(x0, x1, x3), PointToSegmentDistance(x0, x2, x3));
			}
		}
	}

	float GPUSDF::PointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2)
	{
		const glm::vec3 dx(x2 - x1);
		const double m2 = glm::length(dx);
		float s12 = static_cast<float>(dot(x2 - x0, dx) / m2);

		if (s12 < 0.0f) {
			s12 = 0.0f;
		}
		else if (s12 > 1.0f) {
			s12 = 1.0f;
		}

		return glm::distance(x0, s12 * x1 + (1.0f - s12) * x2);
	}

	bool GPUSDF::PointInTriangle2D(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3, double& a, double& b, double& c)
	{
		x1 -= x0;
		x2 -= x0;
		x3 -= x0;
		y1 -= y0;
		y2 -= y0; 
		y3 -= y0;

		const int signa = Orientation(x2, y2, x3, y3, a);
		if (signa == 0) {
			return false;
		}

		const int signb = Orientation(x3, y3, x1, y1, b);
		if (signb != signa) {
			return false;
		}

		const int signc = Orientation(x1, y1, x2, y2, c);
		if (signc != signa) {
			return false;
		}

		const double sum = a + b + c;

		a /= sum;
		b /= sum;
		c /= sum;

		return true;
	}

	int GPUSDF::Orientation(double x1, double y1, double x2, double y2, double& twice_signed_area)
	{
		twice_signed_area = y1 * x2 - x1 * y2;

		if (twice_signed_area > 0.0) {
			return 1;
		}
		else if (twice_signed_area < 0.0) {
			return -1;
		}
		else if (y2 > y1) {
			return 1;
		}
		else if (y2 < y1) {
			return -1;
		}
		else if (x1 > x2) {
			return 1;
		}
		else if (x1 < x2) {
			return -1;
		}
		else {
			return 0;
		}
	}

	void GPUSDF::Sweep(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3D<float>& phi, Array3D<int>& closest_tri, const glm::vec3& origin, float dx, int di, int dj, int dk)
	{
		int i0;
		int i1;

		if (di > 0) {
			i0 = 1; 
			i1 = phi.GetSizeX();
		}
		else {
			i0 = phi.GetSizeX() - 2;
			i1 = -1;
		}

		int j0;
		int j1;

		if (dj > 0) { 
			j0 = 1;
			j1 = phi.GetSizeY();
		}
		else { 
			j0 = phi.GetSizeY() - 2;
			j1 = -1;
		}

		int k0;
		int k1;

		if (dk > 0) {
			k0 = 1; 
			k1 = phi.GetSizeZ();
		}
		else {
			k0 = phi.GetSizeZ() - 2;
			k1 = -1; 
		}

		for (int k = k0; k != k1; k += dk) {
			for (int j = j0; j != j1; j += dj) {
				for (int i = i0; i != i1; i += di) {
					const glm::vec3 gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);

					CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j     , k     );
					CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i     , j - dj, k     );
					CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k     );
					CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i     , j     , k - dk);
					CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j     , k - dk);
					CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i     , j - dj, k - dk);
					CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k - dk);
				}
			}
		}
	}

	void GPUSDF::CheckNeighbor(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3D<float>& phi, Array3D<int>& closest_tri, const glm::vec3& gx, int i0, int j0, int k0, int i1, int j1, int k1)
	{
		if (closest_tri(i1, j1, k1) >= 0) {
			const glm::vec3 v = tri[closest_tri(i1, j1, k1)];
			const unsigned int p = v.x;
			const unsigned int q = v.y;
			const unsigned int r = v.z;

			const float d = PointToTriangleDistance(gx, x[p], x[q], x[r]);

			if (d < phi(i0, j0, k0)) {
				phi(i0, j0, k0) = d;
				closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
			}
		}
	}
}