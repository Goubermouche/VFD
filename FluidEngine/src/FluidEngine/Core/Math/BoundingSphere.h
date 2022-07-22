#ifndef BOUNDING_SPHERE_H_
#define BOUNDING_SPHERE_H_

#include "pch.h"
#include "FluidEngine/Core/Math/Math.h"
#include "FluidEngine/Core/Tree.h"

namespace fe {
	class BoundingSphere
	{
	public:
		BoundingSphere()
			: center({ 0.0f, 0.0f, 0.0f }), radius(0.0f)
		{}

		BoundingSphere(const glm::vec3& x, float r)
			: center(x), radius(r)
		{}

		BoundingSphere(const glm::vec3& a)
			: center(a), radius(0.0f)
		{}

		/// <summary>
		/// Creates an enclosing sphere for 2 points.
		/// </summary>
		/// <param name="a">Point A.</param>
		/// <param name="b">Point B.</param>
		BoundingSphere(const glm::vec3& a, const glm::vec3& b)
		{
			const glm::vec3 ba = b - a;

			center = (a + b) * 0.5f;
			radius = 0.5f * glm::sqrt(glm::dot(ba, ba));
		}

		/// <summary>
		/// Creates an enclosing sphere for 3 points.
		/// </summary>
		/// <param name="a">Point A.</param>
		/// <param name="b">Point B.</param>
		/// <param name="c">Point C.</param>
		BoundingSphere(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
		{
			const glm::vec3 ba = b - a;
			const glm::vec3 ca = c - a;
			const glm::vec3 baxca = glm::cross(ba, ca);

			glm::vec3 r;
			glm::mat3x3 T;

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

			r[0] = 0.5f * glm::dot(ba, ba);
			r[1] = 0.5f * glm::dot(ca, ca);
			r[2] = 0.0f;

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
		BoundingSphere(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d)
		{
			const glm::vec3 ba = b - a;
			const glm::vec3 ca = c - a;
			const glm::vec3 da = d - a;

			glm::vec3 r;
			glm::mat3x3 T;

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

			r[0] = 0.5f * glm::dot(ba, ba);
			r[1] = 0.5f * glm::dot(ca, ca);
			r[2] = 0.5f * glm::dot(da, da);

			center = glm::inverse(T) * r;
			radius = glm::sqrt(glm::dot(center, center));
			center += a;
		}

		/// <summary>
		/// Creates an enclosing sphere for the specified list of points.
		/// </summary>
		/// <param name="p">Set of points.</param>
		BoundingSphere(const std::vector<glm::vec3>& p)
			: center({ 0.0f, 0.0f, 0.0f }), radius(0.0f)
		{
			SetPoints(p);
		}

		/// <summary>
		/// Creates an enclosing sphere for the specified list of points.
		/// </summary>
		/// <param name="p">Set of points.</param>
		void SetPoints(const std::vector<glm::vec3>& p)
		{
			// Remove duplicates
			std::vector<glm::vec3> v(p);
			std::sort(v.begin(), v.end(), [](const glm::vec3& a, const glm::vec3& b)
				{
					if (a.x < b.x) { return true; }
					if (a.x > b.x) { return false; }
					if (a.y < b.y) { return true; }
					if (a.y > b.y) { return false; }
					return (a.z < b.z);
				});

			v.erase(std::unique(v.begin(), v.end(), [](glm::vec3& a, glm::vec3& b) {
				return IsApprox(a, b);
				}), v.end());

			glm::vec3 d;
			const int n = int(v.size());

			// Generate random permutations of the points and perturb the points by epsilon to avoid corner cases
			const float epsilon = 1.0e-6f;
			for (int i = n - 1; i > 0; i--)
			{
				const glm::vec3 epsilon_vec = epsilon * RandomVec3();
				const int j = static_cast<int>(floor(i * float(rand()) / RAND_MAX));
				d = v[i] + epsilon_vec;
				v[i] = v[j] - epsilon_vec;
				v[j] = d;
			}

			BoundingSphere S = BoundingSphere(v[0], v[1]);

			for (int i = 2; i < n; i++)
			{
				//SES0
				d = v[i] - S.center;
				if (glm::dot(d, d) > S.radius * S.radius) {
					S = CalculateSmallestEnclosingSphere(i, v, v[i]);
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
		bool Contains(glm::vec3 const& p) const
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
		/// <returns></returns>
		BoundingSphere CalculateSmallestEnclosingSphere(int n, std::vector<glm::vec3>& p, glm::vec3& q1, glm::vec3& q2, glm::vec3& q3)
		{
			BoundingSphere S(q1, q2, q3);

			for (int i = 0; i < n; i++)
			{
				glm::vec3 d = p[i] - S.center;
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
		/// <returns></returns>
		BoundingSphere CalculateSmallestEnclosingSphere(int n, std::vector<glm::vec3>& p, glm::vec3& q1, glm::vec3& q2)
		{
			BoundingSphere S(q1, q2);

			for (int i = 0; i < n; i++)
			{
				glm::vec3 d = p[i] - S.center;
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
		/// <returns></returns>
		BoundingSphere CalculateSmallestEnclosingSphere(int n, std::vector<glm::vec3>& p, glm::vec3& q1)
		{
			BoundingSphere S(p[0], q1);

			for (int i = 1; i < n; i++)
			{
				glm::vec3 d = p[i] - S.center;
				if (glm::dot(d, d) > S.radius * S.radius) {
					S = CalculateSmallestEnclosingSphere(i, p, q1, p[i]);
				}
			}
			return S;
		}
	public:
		glm::vec3 center;
		float radius;
	};

	class MeshBoundingSphereHierarchy : public Tree<BoundingSphere> {
	public:
		MeshBoundingSphereHierarchy(const std::vector<glm::vec3>& vertices, const std::vector<glm::ivec3>& faces);

		const glm::vec3& GetEntityPosition(uint32_t i) const final;
		void Calculate(uint32_t b, uint32_t n, BoundingSphere& hull) const final;
	private:
		const std::vector<glm::vec3>& m_Vertices;
		const std::vector<glm::ivec3>& m_Faces;
		std::vector<glm::vec3> m_TriangleCenters;
	};
}

#endif // !BOUNDING_SPHERE_H_
