#include "pch.h"
#include "GPUSDF.h"

#include "Compute/Utility/CUDA/cutil_math.h"
#include "GPUSDF.cuh"
//#include "stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

namespace vfd {
	GPUSDF::GPUSDF(Ref<TriangleMesh>& mesh)
	{
		const std::vector<glm::vec3>& vertices = mesh->GetVertices();
		const std::vector<glm::uvec3>& triangles = mesh->GetTriangles();

		m_Domain = BoundingBox(vertices);
		m_Domain.min -= padding * m_CellSize;
		m_Domain.max += padding * m_CellSize;

		m_CellSizeInverse = 1.0f / m_CellSize;
		m_CellCount = glm::compMul(m_Resolution);
		m_Resolution = glm::ceil((m_Domain.max - m_Domain.min) / m_CellSize);

		m_PHI.resize(m_Resolution.x, m_Resolution.y, m_Resolution.z);
		m_PHI.assign((m_Resolution.x + m_Resolution.y + m_Resolution.z) * m_CellSize);

		Array3i closest_tri(m_Resolution.x, m_Resolution.y, m_Resolution.z, -1);
		Array3i intersection_count(m_Resolution.x, m_Resolution.y, m_Resolution.z, 0); // intersection_count(i,j,k) is # of tri intersections in (i-1,i]x{j}x{k}
		// we begin by initializing distances near the mesh, and figuring out intersection counts
		glm::vec3 ijkmin, ijkmax;

		for (unsigned int t = 0; t < triangles.size(); ++t) {
			unsigned int p, q, r;
			p = triangles[t].x;
			q = triangles[t].y;
			r = triangles[t].z;

			// coordinates in grid to high precision
			double fip = ((double)vertices[p][0] - m_Domain.min[0]) / m_CellSize, fjp = ((double)vertices[p][1] - m_Domain.min[1]) / m_CellSize, fkp = ((double)vertices[p][2] - m_Domain.min[2]) / m_CellSize;
			double fiq = ((double)vertices[q][0] - m_Domain.min[0]) / m_CellSize, fjq = ((double)vertices[q][1] - m_Domain.min[1]) / m_CellSize, fkq = ((double)vertices[q][2] - m_Domain.min[2]) / m_CellSize;
			double fir = ((double)vertices[r][0] - m_Domain.min[0]) / m_CellSize, fjr = ((double)vertices[r][1] - m_Domain.min[1]) / m_CellSize, fkr = ((double)vertices[r][2] - m_Domain.min[2]) / m_CellSize;
			// do distances nearby
			int i0 = clamp(int(std::min(std::min(fip, fiq), fir)), 0, m_Resolution.x - 1), i1 = clamp(int(std::max(std::max(fip, fiq), fir)) + 1, 0, m_Resolution.x - 1);
			int j0 = clamp(int(std::min(std::min(fjp, fjq), fjr)), 0, m_Resolution.y - 1), j1 = clamp(int(std::max(std::max(fjp, fjq), fjr)) + 1, 0, m_Resolution.y - 1);
			int k0 = clamp(int(std::min(std::min(fkp, fkq), fkr)), 0, m_Resolution.z - 1), k1 = clamp(int(std::max(std::max(fkp, fkq), fkr)) + 1, 0, m_Resolution.z - 1);
			for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
				glm::vec3 gx(i * m_CellSize + m_Domain.min[0], j * m_CellSize + m_Domain.min[1], k * m_CellSize + m_Domain.min[2]);
				float d = PointToTriangleDistance(gx, vertices[p], vertices[q], vertices[r]);
				if (d < m_PHI(i, j, k)) {
					m_PHI(i, j, k) = d;
					closest_tri(i, j, k) = t;
				}
			}
			// and do intersection counts
			j0 = clamp((int)std::ceil(std::min(std::min(fjp, fjq), fjr)), 0, m_Resolution.y - 1);
			j1 = clamp((int)std::floor(std::max(std::max(fjp, fjq), fjr)), 0, m_Resolution.y - 1);
			k0 = clamp((int)std::ceil(std::min(std::min(fkp, fkq), fkr)), 0, m_Resolution.z - 1);
			k1 = clamp((int)std::floor(std::max(std::max(fkp, fkq), fkr)), 0, m_Resolution.z - 1);
			for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) {
				double a, b, c;
				if (PointInTriangle2D(j, k, fjp, fkp, fjq, fkq, fjr, fkr, a, b, c)) {
					double fi = a * fip + b * fiq + c * fir; // intersection i coordinate
					int i_interval = int(std::ceil(fi)); // intersection is in (i_interval-1,i_interval]
					if (i_interval < 0) ++intersection_count(0, j, k); // we enlarge the first interval to include everything to the -x direction
					else if (i_interval < m_Resolution.x) ++intersection_count(i_interval, j, k);
					// we ignore intersections that are beyond the +x side of the grid
				}
			}
		}

		// and now we fill in the rest of the distances with fast sweeping
		for (unsigned int pass = 0; pass < 2; ++pass) {
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, +1, +1, +1);
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, -1, -1, -1);
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, +1, +1, -1);
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, -1, -1, +1);
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, +1, -1, +1);
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, -1, +1, -1);
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, +1, -1, -1);
			Sweep(triangles, vertices, m_PHI, closest_tri, m_Domain.min, m_CellSize, -1, +1, +1);
		}

		// then figure out signs (inside/outside) from intersection counts
		for (int k = 0; k < m_Resolution.z; ++k) for (int j = 0; j < m_Resolution.y; ++j) {
			int total_count = 0;
			for (int i = 0; i < m_Resolution.x; ++i) {
				total_count += intersection_count(i, j, k);
				if (total_count % 2 == 1) { // if parity of intersections so far is odd,
					m_PHI(i, j, k) = -m_PHI(i, j, k); // we are inside the mesh
				}
			}
		}
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

		if (m_PHI.indexInRange(index.x, index.y, index.z)) {
			c000 = m_PHI(index.x, index.y, index.z);
		}

		if (m_PHI.indexInRange(index.x + 1, index.y, index.z)) {
			c100 = m_PHI(index.x + 1, index.y, index.z);
		}

		if (m_PHI.indexInRange(index.x, index.y + 1, index.z)) {
			c010 = m_PHI(index.x, index.y + 1, index.z);
		}

		if (m_PHI.indexInRange(index.x + 1, index.y + 1, index.z)) {
			c110 = m_PHI(index.x + 1, index.y + 1, index.z);
		}

		if (m_PHI.indexInRange(index.x, index.y, index.z + 1)) {
			c001 = m_PHI(index.x, index.y, index.z + 1);
		}

		if (m_PHI.indexInRange(index.x + 1, index.y, index.z + 1)) {
			c101 = m_PHI(index.x + 1, index.y, index.z + 1);
		}

		if (m_PHI.indexInRange(index.x, index.y + 1, index.z + 1)) {
			c011 = m_PHI(index.x, index.y + 1, index.z + 1);
		}

		if (m_PHI.indexInRange(index.x + 1, index.y + 1, index.z + 1)) {
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

		glm::mat4x4 weights(
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
		// first find barycentric coordinates of closest point on infinite plane
		glm::vec3 x13(x1 - x3), x23(x2 - x3), x03(x0 - x3);
		float m13 = glm::length(x13), m23 = glm::length(x23), d = dot(x13, x23);
		float invdet = 1.f / max(m13 * m23 - d * d, 1e-30f);
		float a = dot(x13, x03), b = dot(x23, x03);
		// the barycentric coordinates themselves
		float w23 = invdet * (m23 * a - d * b);
		float w31 = invdet * (m13 * b - d * a);
		float w12 = 1 - w23 - w31;
		if (w23 >= 0 && w31 >= 0 && w12 >= 0) {  // if we're inside the triangle
			return glm::distance(x0, w23 * x1 + w31 * x2 + w12 * x3);
		}
		else {        // we have to clamp to one of the edges
			if (w23 > 0) {
				return std::min(PointToSegmentDistance(x0, x1, x2), PointToSegmentDistance(x0, x1, x3));
			}
			else if (w31 > 0) {
				return std::min(PointToSegmentDistance(x0, x1, x2), PointToSegmentDistance(x0, x2, x3));
			}
			else {
				return std::min(PointToSegmentDistance(x0, x1, x3), PointToSegmentDistance(x0, x2, x3));
			}
		}
	}

	float GPUSDF::PointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2)
	{
		glm::vec3 dx(x2 - x1);
		double m2 = glm::length(dx);
		// find parameter value of closest point on segment
		float s12 = (float)(dot(x2 - x0, dx) / m2);
		if (s12 < 0) {
			s12 = 0;
		}
		else if (s12 > 1) {
			s12 = 1;
		}

		// and find the distance
		float d = glm::distance(x0, s12 * x1 + (1 - s12) * x2);
		return d;
	}

	bool GPUSDF::PointInTriangle2D(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3, double& a, double& b, double& c)
	{
		x1 -= x0; x2 -= x0; x3 -= x0;
		y1 -= y0; y2 -= y0; y3 -= y0;
		int signa = Orientation(x2, y2, x3, y3, a);
		if (signa == 0) return false;
		int signb = Orientation(x3, y3, x1, y1, b);
		if (signb != signa) return false;
		int signc = Orientation(x1, y1, x2, y2, c);
		if (signc != signa) return false;
		double sum = a + b + c;
		assert(sum != 0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
		a /= sum;
		b /= sum;
		c /= sum;
		return true;
	}

	int GPUSDF::Orientation(double x1, double y1, double x2, double y2, double& twice_signed_area)
	{
		twice_signed_area = y1 * x2 - x1 * y2;
		if (twice_signed_area > 0) return 1;
		else if (twice_signed_area < 0) return -1;
		else if (y2 > y1) return 1;
		else if (y2 < y1) return -1;
		else if (x1 > x2) return 1;
		else if (x1 < x2) return -1;
		else return 0; // only true when x1==x2 and y1==y2
	}

	void GPUSDF::Sweep(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3f& phi, Array3i& closest_tri, const glm::vec3& origin, float dx, int di, int dj, int dk)
	{
		int i0, i1;
		if (di > 0) { i0 = 1; i1 = phi.ni; }
		else { i0 = phi.ni - 2; i1 = -1; }
		int j0, j1;
		if (dj > 0) { j0 = 1; j1 = phi.nj; }
		else { j0 = phi.nj - 2; j1 = -1; }
		int k0, k1;
		if (dk > 0) { k0 = 1; k1 = phi.nk; }
		else { k0 = phi.nk - 2; k1 = -1; }
		for (int k = k0; k != k1; k += dk) for (int j = j0; j != j1; j += dj) for (int i = i0; i != i1; i += di) {
			glm::vec3 gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);
			CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k);
			CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k);
			CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k);
			CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i, j, k - dk);
			CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k - dk);
			CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k - dk);
			CheckNeighbor(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k - dk);
		}
	}

	void GPUSDF::CheckNeighbor(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3f& phi, Array3i& closest_tri, const glm::vec3& gx, int i0, int j0, int k0, int i1, int j1, int k1)
	{
		if (closest_tri(i1, j1, k1) >= 0) {

			auto v = tri[closest_tri(i1, j1, k1)];
			unsigned int p = v.x;
			unsigned int q = v.y;
			unsigned int r = v.z;

			float d = PointToTriangleDistance(gx, x[p], x[q], x[r]);
			if (d < phi(i0, j0, k0)) {
				phi(i0, j0, k0) = d;
				closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
			}
		}
	}
}