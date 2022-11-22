#ifndef BOUNDING_SPHERE_H
#define BOUNDING_SPHERE_H

#include "pch.h"
#include "Core/Math/Math.h"
#include "Core/Structures/Tree.h"

namespace vfd {
	class BoundingSphere
	{
	public:
		BoundingSphere() = default;

		BoundingSphere(const glm::dvec3& x, float r)
			: center(x), radius(r)
		{}

		BoundingSphere(const glm::dvec3& a)
			: center(a), radius(0.0f)
		{}

		/// <summary>
		/// Creates an enclosing sphere for 2 points.
		/// </summary>
		/// <param name="a">Point A.</param>
		/// <param name="b">Point B.</param>
		BoundingSphere(const glm::dvec3& a, const glm::dvec3& b)
		{
			const glm::dvec3 ba = b - a;

			center = (a + b) * 0.5;
			radius = 0.5 * glm::sqrt(glm::dot(ba, ba));
		}

		/// <summary>
		/// Creates an enclosing sphere for 3 points.
		/// </summary>
		/// <param name="a">Point A.</param>
		/// <param name="b">Point B.</param>
		/// <param name="c">Point C.</param>
		BoundingSphere(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c)
		{
			const glm::dvec3 ba = b - a;
			const glm::dvec3 ca = c - a;
			const glm::dvec3 baxca = glm::cross(ba, ca);

			glm::dvec3 r;
			glm::dmat3x3 T;

			// glm::row(T, 0) = ba;
			// glm::row(T, 1) = ca;
			// glm::row(T, 2) = baxca;

			T[0][0] = ba[0];
			T[1][0] = ba[1];
			T[2][0] = ba[2];

			T[0][1] = ca[0];
			T[1][1] = ca[1];
			T[2][1] = ca[2];

			T[0][2] = baxca[0];
			T[1][2] = baxca[1];
			T[2][2] = baxca[2];

			r[0] = 0.5 * glm::dot(ba, ba);
			r[1] = 0.5 * glm::dot(ca, ca);
			r[2] = 0.0;

			center = glm::inverse(T) * r;
			radius = glm::sqrt(glm::dot(center, center));
			center += a;
		}

		/// <summary>
		/// Creates an enclosing sphere for 4 points.
		/// </summary>
		/// <param name="a">Point A.</param>
		/// <param name="b">Point B.</param>
		/// <param name="c">Point C.</param>
		/// <param name="d">Point D.</param>
		BoundingSphere(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c, const glm::dvec3& d)
		{
			const glm::dvec3 ba = b - a;
			const glm::dvec3 ca = c - a;
			const glm::dvec3 da = d - a;

			glm::dvec3 r;
			glm::dmat3x3 T;

			// glm::row(T, 0) = ba;
			// glm::row(T, 1) = ca;
			// glm::row(T, 2) = da;

			T[0][0] = ba[0];
			T[1][0] = ba[1];
			T[2][0] = ba[2];

			T[0][1] = ca[0];
			T[1][1] = ca[1];
			T[2][1] = ca[2];

			T[0][2] = da[0];
			T[1][2] = da[1];
			T[2][2] = da[2];

			r[0] = 0.5 * glm::dot(ba, ba);
			r[1] = 0.5 * glm::dot(ca, ca);
			r[2] = 0.5 * glm::dot(da, da);

			center = glm::inverse(T) * r;
			radius = glm::sqrt(glm::dot(center, center));
			center += a;
		}

		/// <summary>
		/// Creates an enclosing sphere for the specified list of points.
		/// </summary>
		/// <param name="p">Set of points.</param>
		BoundingSphere(const std::vector<glm::dvec3>& p)
			: center({ 0.0, 0.0, 0.0 }), radius(0.0)
		{
			SetPoints(p);
		}

		/// <summary>
		/// Creates an enclosing sphere for the specified list of points.
		/// </summary>
		/// <param name="points">Set of points.</param>
		void SetPoints(const std::vector<glm::dvec3>& points)
		{
			// Remove duplicates
			std::vector<glm::dvec3> vertices(points);
			std::ranges::sort(vertices.begin(), vertices.end(), [](const glm::dvec3& a, const glm::dvec3& b)
			{
				if (a.x < b.x) { return true; }
				if (a.x > b.x) { return false; }
				if (a.y < b.y) { return true; }
				if (a.y > b.y) { return false; }
				return (a.z < b.z);
			});

			vertices.erase(std::unique(vertices.begin(), vertices.end(), [](const glm::dvec3& a, const glm::dvec3& b) {
				return IsApprox(a, b);
			}), vertices.end());

			glm::dvec3 d;
			const unsigned int n = vertices.size();

			// Generate random permutations of the points and perturb the points by epsilon to avoid corner cases
			constexpr float epsilon = 1.0e-6f;
			for (int i = n - 1; i > 0; i--)
			{
				const glm::dvec3 epsilonVec = epsilon * Random::Vec3(-1.0f, 1.0f);
				const int j = static_cast<int>(floor(i * rand()) / RAND_MAX);
				d = vertices[i] + epsilonVec;
				vertices[i] = vertices[j] - epsilonVec;
				vertices[j] = d;
			}

			BoundingSphere S = BoundingSphere(vertices[0], vertices[1]);

			for (int i = 2; i < n; i++)
			{
				//SES0
				d = vertices[i] - S.center;
				if (glm::dot(d, d) > S.radius * S.radius) {
					S = CalculateSmallestEnclosingSphere(i, vertices, vertices[i]);
				}
			}

			center = S.center;
			radius = S.radius + epsilon; // Add epsilon to make sure that all non-perturbed points are inside the sphere
		}

		/// <summary>
		/// Checks if two spheres overlap
		/// </summary>
		/// <param name="other">Second sphere.</param>
		/// <returns>Intersection test result.</returns>
		[[nodiscard]]
		bool Overlaps(BoundingSphere const& other) const
		{
			const float rr = radius + other.radius;
			return glm::dot((center - other.center), (center - other.center)) < rr * rr;
		}

		/// <summary>
		/// Checks if a given sphere is fully contained inside another one.
		/// </summary>
		/// <param name="other">Sphere to contain.</param>
		/// <returns>Containment test result.</returns>
		[[nodiscard]]
		bool Contains(BoundingSphere const& other) const
		{
			const float rr = radius - other.radius;
			return glm::dot((center - other.center), (center - other.center)) < rr * rr;
		}

		/// <summary>
		/// Chekcs if the sphere contains the specified point.
		/// </summary>
		/// <param name="p">Point to check.</param>
		/// <returns>Containment test result.</returns>
		[[nodiscard]]
		bool Contains(glm::dvec3 const& p) const
		{
			return glm::dot((center - p), (center - p)) < radius * radius;
		}
	private:
		/// <summary>
		/// Creates the smallest enclosing sphere for n points with the points q1, q2, and q3 on the surface of the sphere.
		/// </summary>
		/// <param name="n">Number of points to contain.</param>
		/// <param name="p">List of points to contain</param>
		/// <param name="q1">Point on a surface.</param>
		/// <param name="q2">Point on a surface.</param>
		/// <param name="q3">Point on a surface.</param>
		/// <returns>The smallest enclosing sphere.</returns>
		[[nodiscard]]
		BoundingSphere CalculateSmallestEnclosingSphere(const int n, const std::vector<glm::dvec3>& p,const glm::dvec3& q1,const glm::dvec3& q2,const glm::dvec3& q3) const
		{
			BoundingSphere S(q1, q2, q3);

			for (int i = 0; i < n; i++)
			{
				glm::dvec3 d = p[i] - S.center;
				if (glm::dot(d, d) > S.radius * S.radius) {
					S = BoundingSphere(q1, q2, q3, p[i]);
				}
			}
			return S;
		}

		/// <summary>
		/// Creates the smallest enclosing sphere for n points with the points q1 and q2 on the surface of the sphere.
		/// </summary>
		/// <param name="n">Number of points to contain.</param>
		/// <param name="p">List of points to contain</param>
		/// <param name="q1">Point on a surface.</param>
		/// <param name="q2">Point on a surface.</param>
		/// <returns>The smallest enclosing sphere.</returns>
		[[nodiscard]]
		BoundingSphere CalculateSmallestEnclosingSphere(const int n, const std::vector<glm::dvec3>& p, const glm::dvec3& q1, const glm::dvec3& q2) const
		{
			BoundingSphere S(q1, q2);

			for (int i = 0; i < n; i++)
			{
				glm::dvec3 d = p[i] - S.center;
				if (glm::dot(d, d) > S.radius * S.radius) {
					S = CalculateSmallestEnclosingSphere(i, p, q1, q2, p[i]);
				}
			}
			return S;
		}

		/// <summary>
		/// Creates the smallest enclosing sphere for n points with the point q1 on the surface of the sphere.
		/// </summary>
		/// <param name="n">Number of points to contain.</param>
		/// <param name="p">List of points to contain</param>
		/// <param name="q1">Point on a surface.</param>
		/// <returns>The smallest enclosing sphere.</returns>
		[[nodiscard]]
		BoundingSphere CalculateSmallestEnclosingSphere(const int n, const std::vector<glm::dvec3>& p, const glm::dvec3& q1) const
		{
			BoundingSphere S(p[0], q1);

			for (int i = 1; i < n; i++)
			{
				glm::dvec3 d = p[i] - S.center;
				if (glm::dot(d, d) > S.radius * S.radius) {
					S = CalculateSmallestEnclosingSphere(i, p, q1, p[i]);
				}
			}
			return S;
		}
	public:
		glm::dvec3 center = { 0.0, 0.0, 0.0 };
		double radius = 0.0;
	};

	class MeshBoundingSphereHierarchy : public Tree<BoundingSphere>, public RefCounted {
	public:
		MeshBoundingSphereHierarchy(const std::vector<glm::dvec3>& vertices, const std::vector<glm::uvec3>& faces);

		const glm::dvec3& GetEntityPosition(unsigned int i) const final;
		void Calculate(unsigned int b, unsigned int n, BoundingSphere& hull) const final;
	private:
		const std::vector<glm::dvec3>& m_Vertices;
		const std::vector<glm::uvec3>& m_Faces;
		std::vector<glm::dvec3> m_TriangleCenters;
	};
}

#endif // !BOUNDING_SPHERE_H
