#include "pch.h"
#include "MeshDistance.h"
#include <omp.h>

namespace glm {
	bool operator<(const glm::vec3& lhs, const glm::vec3& rhs)
	{
		return lhs.x < rhs.x || lhs.x == rhs.x && (lhs.y < rhs.y || lhs.y == rhs.y && lhs.z < rhs.z);
	}
}

namespace fe {
	MeshDistance::MeshDistance(const EdgeMesh& mesh, bool preCalculateNormals)
		: m_Mesh(mesh), m_BSH(mesh.GetVertices(), mesh.GetFaces()), m_PreCalculatedNormals(preCalculateNormals)
	{
		const uint32_t maxThreads = omp_get_max_threads();
		m_Queues.resize(maxThreads);
		m_ClosestFace.resize(maxThreads);
		m_Cache.resize(maxThreads, Cache<glm::vec3, float>([&](glm::vec3 const& xi) {
			return SignedDistance(xi);
		}, 10000));

		m_BSH.Construct();

		if (m_PreCalculatedNormals)
		{
			m_FaceNormals.resize(m_Mesh.GetFaceCount());
			m_VertexNormals.resize(mesh.GetVertexCount(), { 0.0f, 0.0f, 0.0f });

			uint32_t index = 0;
			for (const glm::ivec3 face : m_Mesh.GetFaces())
			{
				glm::vec3 const& x0 = m_Mesh.GetVertex(face.x);
				glm::vec3 const& x1 = m_Mesh.GetVertex(face.y);
				glm::vec3 const& x2 = m_Mesh.GetVertex(face.z);

				glm::vec3 n = glm::normalize(glm::cross(x1 - x0, x2 - x0));

				glm::vec3 e1 = glm::normalize(x1 - x0);
				glm::vec3 e2 = glm::normalize(x2 - x1);
				glm::vec3 e3 = glm::normalize(x0 - x2);

				const glm::vec3 alpha = glm::vec3{
					  std::acos(glm::dot(e1, -e3)),
					  std::acos(glm::dot(e1, -e1)),
					  std::acos(glm::dot(e1, -e2))
				};

				m_VertexNormals[face.x] += alpha.x * n;
				m_VertexNormals[face.y] += alpha.y * n;
				m_VertexNormals[face.z] += alpha.z * n;

				m_FaceNormals[index] = n;
				index++;
			}
		}
	}

	float MeshDistance::Distance(const glm::vec3& point, glm::vec3* closestPoint, uint32_t* closestFace, Triangle* closestEntity) const
	{
		using namespace std::placeholders;

		float distanceCandidate = std::numeric_limits<float>::max();
		auto face = m_ClosestFace[omp_get_thread_num()];

		if (face < m_Mesh.GetFaceCount())
		{
			auto t = std::array<glm::vec3 const*, 3>{
				&m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 0)),
					& m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 1)),
					& m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 2))
			};
			distanceCandidate = std::sqrt(PointTriangleDistanceSquared(point, t));
		}

		auto predicate = [&](const uint32_t nodeIndex, uint32_t)
		{
			return Predicate(nodeIndex, m_BSH, point, distanceCandidate);
		};

		auto callback = [&](const uint32_t nodeIndex, uint32_t)
		{
			return Callback(nodeIndex, point, distanceCandidate);
		};

		while (!m_Queues[omp_get_thread_num()].empty()) {
			m_Queues[omp_get_thread_num()].pop();
		}

		m_BSH.TraverseDepthFirst(predicate, callback);

		face = m_ClosestFace[omp_get_thread_num()];
		if (closestPoint)
		{
			auto t = std::array<glm::vec3 const*, 3>{
				&m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 0)),
					& m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 1)),
					& m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 2))
			};

			glm::vec3 np;
			Triangle ne = Triangle{};
			const float dist2 = PointTriangleDistanceSquared(point, t, &np, &ne);
			distanceCandidate = std::sqrt(dist2);

			if (closestEntity) {
				*closestEntity = ne;
			}
			if (closestPoint) {
				*closestPoint = np;
			}
		}

		if (closestFace) {
			*closestFace = face;
		}

		return distanceCandidate;
	}

	void MeshDistance::Callback(const uint32_t nodeIndex, const glm::vec3& point, float& distanceCandidate) const
	{
		auto const& node = m_BSH.GetNode(nodeIndex);
		auto const& hull = m_BSH.GetType(nodeIndex);

		if (!node.IsLeaf()) {
			return;
		}

		const float radius = hull.radius;

		const glm::vec3 temp = (point - hull.center);
		const float distanceToCenter = temp.x * temp.x + temp.y * temp.y + temp.z * temp.z;

		const float distanceRadiusCandidate = distanceCandidate + radius;
		if (distanceToCenter > distanceRadiusCandidate * distanceRadiusCandidate) {
			return;
		}

		float distanceCandidate2 = distanceCandidate * distanceCandidate;
		bool changed = false;
		for (uint32_t i = node.begin; i < node.begin + node.n; ++i)
		{
			const uint32_t f = m_BSH.GetEntity(i);

			auto t = std::array<glm::vec3 const*, 3>{
				&m_Mesh.GetVertex(m_Mesh.GetFaceVertex(f, 0)),
					& m_Mesh.GetVertex(m_Mesh.GetFaceVertex(f, 1)),
					& m_Mesh.GetVertex(m_Mesh.GetFaceVertex(f, 2))
			};

			const float dist2_ = PointTriangleDistanceSquared(point, t);
			if (distanceCandidate2 > dist2_)
			{
				distanceCandidate2 = dist2_;
				changed = true;
				m_ClosestFace[omp_get_thread_num()] = f;
			}
		}

		if (changed)
		{
			distanceCandidate = std::sqrt(distanceCandidate2);
		}
	}

	bool MeshDistance::Predicate(const uint32_t nodeIndex, const MeshBoundingSphereHierarchy& bsh, const glm::vec3& point, float& distanceCandidate) const
	{
		// If the furthest point on the current candidate hull is closer than the closest point on the next hull then we can skip it
		const BoundingSphere& hull = bsh.GetType(nodeIndex);
		const float& hullRadius = hull.radius;
		const glm::vec3& hullCenter = hull.center;

		const auto distanceToCenterSquared = glm::dot((point - hullCenter), (point - hullCenter));

		if (distanceCandidate > hullRadius) {
			const float l = distanceCandidate - hullRadius;
			if (l * l > distanceToCenterSquared) {
				distanceCandidate = std::sqrt(distanceToCenterSquared) + hullRadius;
			}
		}

		const float d = distanceCandidate + hullRadius;
		return distanceToCenterSquared <= d * d;
	}

	float MeshDistance::SignedDistance(const glm::vec3& point) const
	{
		uint32_t closestFace;
		Triangle closestEntity;
		glm::vec3 closestPoint;
		glm::vec3 normal;
		float distance = Distance(point, &closestPoint, &closestFace, &closestEntity);

		switch (closestEntity)
		{
		case Triangle::A:
			normal = CalculateVertexNormal(m_Mesh.GetFaceVertex(closestFace, 0));
			break;
		case Triangle::B:
			normal = CalculateVertexNormal(m_Mesh.GetFaceVertex(closestFace, 1));
			break;
		case Triangle::C:
			normal = CalculateVertexNormal(m_Mesh.GetFaceVertex(closestFace, 2));
			break;
		case Triangle::D:
			normal = CalculateEdgeNormal({ closestFace, 0 });
			break;
		case Triangle::E:
			normal = CalculateEdgeNormal({ closestFace, 1 });
			break;
		case Triangle::F:
			normal = CalculateEdgeNormal({ closestFace, 2 });
			break;
		case Triangle::G:
			normal = CalculateFaceNormal(closestFace);
			break;
		default:
			normal = { 0.0f, 0.0f, 0.0f };
			break;
		}

		if (glm::dot((point - closestPoint), (normal)) < 0.0f) {
			distance *= -1.0f;
		}

		return distance;
	}

	float MeshDistance::SignedDistanceCached(const glm::vec3& point) const
	{
		return m_Cache[omp_get_thread_num()](point);
	}

	glm::vec3 MeshDistance::CalculateVertexNormal(const uint32_t vertex) const
	{
		if (m_PreCalculatedNormals) {
			return m_VertexNormals[vertex];
		}

		const glm::vec3& x0 = m_Mesh.GetVertex(vertex);
		glm::vec3 normal = { 0.0f, 0.0f, 0.0f };

		for (HalfEdge h : m_Mesh.GetIncidentFaces(vertex))
		{
			assert(m_Mesh.Source(h) == vertex);
			const uint32_t ve0 = m_Mesh.Target(h);
			glm::vec3 e0 = (m_Mesh.GetVertex(ve0) - x0);
			e0 = glm::normalize(e0);
			const uint32_t ve1 = m_Mesh.Target(h.GetNext());
		    glm::vec3 e1 = (m_Mesh.GetVertex(ve1) - x0);
			e0 = glm::normalize(e0);
			const float alpha = std::acos((glm::dot(e0, e1)));
			normal += alpha * glm::cross(e0, e1);
		}

		return normal;
	}

	glm::vec3 MeshDistance::CalculateEdgeNormal(const HalfEdge& halfEdge) const
	{
		const HalfEdge oppositeHalfEdge = m_Mesh.Opposite(halfEdge);

		if (m_PreCalculatedNormals)
		{
			if (oppositeHalfEdge.IsBoundary()) {
				return m_FaceNormals[halfEdge.GetFace()];
			}

			return m_FaceNormals[halfEdge.GetFace()] + m_FaceNormals[oppositeHalfEdge.GetFace()];
		}

		if (oppositeHalfEdge.IsBoundary()) {
			return CalculateFaceNormal(halfEdge.GetFace());
		}

		return CalculateFaceNormal(halfEdge.GetFace()) + CalculateFaceNormal(oppositeHalfEdge.GetFace());
	}

	glm::vec3 MeshDistance::CalculateFaceNormal(const uint32_t face) const
	{
		if (m_PreCalculatedNormals) {
			return m_FaceNormals[face];
		}

		const glm::vec3& x0 = m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 0));
		const glm::vec3& x1 = m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 1));
		const glm::vec3& x2 = m_Mesh.GetVertex(m_Mesh.GetFaceVertex(face, 2));

		return glm::normalize(glm::cross((x1 - x0), (x2 - x0)));
	}

	float MeshDistance::PointTriangleDistanceSquared(const glm::vec3& point, const std::array<glm::vec3 const*, 3>& triangle, glm::vec3* closestPoint, Triangle* closestEntity)
	{
		glm::vec3 diff = *triangle[0] - point;
		glm::vec3 edge0 = *triangle[1] - *triangle[0];
		glm::vec3 edge1 = *triangle[2] - *triangle[0];
		float a00 = glm::dot(edge0, edge0);
		float a01 = glm::dot(edge0, edge1);
		float a11 = glm::dot(edge1, edge1);
		float b0 = glm::dot(diff, edge0);
		float b1 = glm::dot(diff, edge1);
		float c = glm::dot(diff, diff);
		float det = std::abs(a00 * a11 - a01 * a01);
		float s = a01 * b1 - a11 * b0;
		float t = a01 * b0 - a00 * b1;

		float d2 = -1.0f;

		if (s + t <= det)
		{
			if (s < 0.0f)
			{
				if (t < 0.0f)
				{
					if (b0 < 0.0f)
					{
						t = 0.0f;
						if (-b0 >= a00)
						{   // T1
							if (closestEntity) {
								*closestEntity = Triangle::B;
							}
							s = 1.0f;
							d2 = a00 + (2.0f) * b0 + c;
						}
						else
						{
							// T3
							if (closestEntity) {
								*closestEntity = Triangle::D;
							}
							s = -b0 / a00;
							d2 = b0 * s + c;
						}
					}
					else
					{
						s = 0.0f;
						if (b1 >= 0.0f)
						{   // T0
							if (closestEntity) {
								*closestEntity = Triangle::A;
							}
							t = 0.0f;
							d2 = c;
						}
						else if (-b1 >= a11)
						{
							// T2
							if (closestEntity) {
								*closestEntity = Triangle::C;
							}
							t = 1.0f;
							d2 = a11 + (2.0f) * b1 + c;
						}
						else
						{
							// T5
							if (closestEntity) {
								*closestEntity = Triangle::F;
							}
							t = -b1 / a11;
							d2 = b1 * t + c;
						}
					}
				}
				else  // region 3
				{
					s = 0.0f;
					if (b1 >= 0.0f)
					{   // T0
						if (closestEntity) {
							*closestEntity = Triangle::A;
						}
						t = 0.0f;
						d2 = c;
					}
					else if (-b1 >= a11)
					{   // T2
						if (closestEntity) {
							*closestEntity = Triangle::C;
						}
						t = 1.0f;
						d2 = a11 + (2.0f) * b1 + c;
					}
					else
					{   // T5
						if (closestEntity) {
							*closestEntity = Triangle::F;
						}
						t = -b1 / a11;
						d2 = b1 * t + c;
					}
				}
			}
			else if (t < 0.0f)  // region 5
			{
				t = 0.0f;
				if (b0 >= 0.0f)
				{   // T0
					if (closestEntity) {
						*closestEntity = Triangle::A;
					}
					s = 0.0f;
					d2 = c;
				}
				else if (-b0 >= a00)
				{   // T1
					if (closestEntity) {
						*closestEntity = Triangle::B;
					}
					s = 1.0f;
					d2 = a00 + (2.0f) * b0 + c;
				}
				else
				{   // T3
					if (closestEntity) {
						*closestEntity = Triangle::D;
					}
					s = -b0 / a00;
					d2 = b0 * s + c;
				}
			}
			else  // region 0 
			{   // T6
				if (closestEntity) {
					*closestEntity = Triangle::G;
				}
				// minimum at interior point
				float invDet = (1.0f) / det;
				s *= invDet;
				t *= invDet;
				d2 = s * (a00 * s + a01 * t + (2.0f) * b0) + t * (a01 * s + a11 * t + (2.0f) * b1) + c;
			}
		}
		else
		{
			float tmp0;
			float tmp1;
			float numer;
			float denom;

			if (s < 0.0f)  // region 2
			{
				tmp0 = a01 + b0;
				tmp1 = a11 + b1;
				if (tmp1 > tmp0)
				{
					numer = tmp1 - tmp0;
					denom = a00 - (2.0f) * a01 + a11;
					if (numer >= denom)
					{   // T1
						if (closestEntity) {
							*closestEntity = Triangle::B;
						}
						s = 1.0f;
						t = 0.0f;
						d2 = a00 + (2.0f) * b0 + c;
					}
					else
					{
						// T4
						if (closestEntity) {
							*closestEntity = Triangle::E;
						}
						s = numer / denom;
						t = 1 - s;
						d2 = s * (a00 * s + a01 * t + (2.0f) * b0) +
							t * (a01 * s + a11 * t + (2.0f) * b1) + c;
					}
				}
				else
				{
					s = 0.0f;
					if (tmp1 <= 0.0f)
					{   // T2
						if (closestEntity) {
							*closestEntity = Triangle::C;
						}
						t = 1.0f;
						d2 = a11 + (2.0f) * b1 + c;
					}
					else if (b1 >= 0.0f)
					{   // T0
						if (closestEntity) {
							*closestEntity = Triangle::A;
						}
						t = 0.0f;
						d2 = c;
					}
					else
					{
						// T5
						if (closestEntity) {
							*closestEntity = Triangle::F;
						}
						t = -b1 / a11;
						d2 = b1 * t + c;
					}
				}
			}
			else if (t < 0.0f)  // region 6
			{
				tmp0 = a01 + b1;
				tmp1 = a00 + b0;
				if (tmp1 > tmp0)
				{
					numer = tmp1 - tmp0;
					denom = a00 - (2.0f) * a01 + a11;
					if (numer >= denom)
					{   // T2
						if (closestEntity) {
							*closestEntity = Triangle::C;
						}
						t = 1.0f;
						s = 0.0f;
						d2 = a11 + (2.0f) * b1 + c;
					}
					else
					{
						// T4
						if (closestEntity) {
							*closestEntity = Triangle::E;
						}
						t = numer / denom;
						s = 1 - t;
						d2 = s * (a00 * s + a01 * t + (2.0f) * b0) + t * (a01 * s + a11 * t + (2.0f) * b1) + c;
					}
				}
				else
				{
					t = 0.0f;
					if (tmp1 <= 0.0f)
					{   // T1
						if (closestEntity) {
							*closestEntity = Triangle::B;
						}
						s = 1;
						d2 = a00 + (2.0f) * b0 + c;
					}
					else if (b0 >= 0.0f)
					{   // T0
						if (closestEntity) {
							*closestEntity = Triangle::A;
						}
						s = 0.0f;
						d2 = c;
					}
					else
					{
						// T3
						if (closestEntity) {
							*closestEntity = Triangle::D;
						}
						s = -b0 / a00;
						d2 = b0 * s + c;
					}
				}
			}
			else  // region 1
			{
				numer = a11 + b1 - a01 - b0;
				if (numer <= 0.0f)
				{   // T2
					if (closestEntity) {
						*closestEntity = Triangle::C;
					}
					s = 0.0f;
					t = 1.0f;
					d2 = a11 + (2.0f) * b1 + c;
				}
				else
				{
					denom = a00 - (2.0f) * a01 + a11;
					if (numer >= denom)
					{   // T1
						if (closestEntity) {
							*closestEntity = Triangle::B;
						}
						s = 1.0f;
						t = 0.0f;
						d2 = a00 + (2.0f) * b0 + c;
					}
					else
					{   // T4
						if (closestEntity) {
							*closestEntity = Triangle::E;
						}
						s = numer / denom;
						t = 1 - s;
						d2 = s * (a00 * s + a01 * t + (2.0f) * b0) + t * (a01 * s + a11 * t + (2.0f) * b1) + c;
					}
				}
			}
		}

		// Account for numerical round-off error.
		if (d2 < 0.0f) {
			d2 = 0.0f;
		}

		if (closestPoint) {
			*closestPoint = *triangle[0] + s * edge0 + t * edge1;
		}

		return d2;
	}
}