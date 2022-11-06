#ifndef MATH_H
#define MATH_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#define PI 3.14159265358979323846
#define PI2 2.0 * PI
#define EPS 1.0e-5

namespace fe {
	/// <summary>
	/// Decomposes a transform into its base components.
	/// </summary>
	/// <param name="transform">Transform matrix to decompose.</param>
	/// <param name="translation">Out translation.</param>
	/// <param name="rotation">Out rotation.</param>
	/// <param name="scale">Out scale</param>
	/// <returns>Whether the decomposition was successful</returns>
	bool DecomposeTransform(const glm::mat4& transform, glm::vec3& Position, glm::vec3& Rotation, glm::vec3& Scale);

	static void GetOrthogonalVectors(const glm::vec3& vec, glm::vec3& x, glm::vec3& y) {

		// Get plane vectors x, y
		glm::vec3 v(1.0, 0.0, 0.0);

		// Check, if v has same direction as vec
		if (fabs(glm::dot(v, vec)) > 0.999) {
			v = glm::vec3(0.0, 1.0, 0.0);
		}

		x = glm::cross(vec, v);
		y = glm::cross(vec, x);
		x = glm::normalize(x);
		y = glm::normalize(y);
	}

	static void JacobiRotate(glm::mat3x3& A, glm::mat3x3& R, int p, int q) {
		if (A[p][q] == 0.0f) {
			return;
		}

		float d = (A[p][p] - A[q][q]) / (static_cast<float>(2.0) * A[p][q]);
		float t = static_cast<float>(1.0) / (fabs(d) + sqrt(d * d + static_cast<float>(1.0)));

		if (d < 0.0f) {
			t = -t;
		}

		float c = static_cast<float>(1.0) / sqrt(t * t + 1);
		float s = t * c;

		A[p][p] += t * A[p][q];
		A[q][q] -= t * A[p][q];
		A[p][q] = A[q][p] = 0.0f;

		int k;

		for (k = 0; k < 3; k++) {
			if (k != p && k != q) {
				float Akp = c * A[k][p] + s * A[k][q];
				float Akq = -s * A[k][p] + c * A[k][q];
				A[k][p] = A[p][k] = Akp;
				A[k][q] = A[q][k] = Akq;
			}
		}

		for (k = 0; k < 3; k++) {
			float Rkp = c * R[k][p] + s * R[k][q];
			float Rkq = -s * R[k][p] + c * R[k][q];
			R[k][p] = Rkp;
			R[k][q] = Rkq;
		}
	}

	static void EigenDecomposition(const glm::mat3x3& A, glm::mat3x3& eigenVecs, glm::vec3& eigenVals) {
		const int numJacobiIterations = 10;
		const float epsilon = static_cast<float>(1e-15);

		glm::mat3x3 D = A;

		eigenVecs = glm::mat3x3(1.0f);
		int iter = 0;
		while (iter < numJacobiIterations) {
			int p = 0;
			int q = 1;
			float a = fabs(D[0][2]);
			float max = fabs(D[0][1]);

			if (a > max) { 
				p = 0;
				q = 2;
				max = a;
			}

			a = fabs(D[1][2]);

			if (a > max) { 
				p = 1; 
				q = 2; 
				max = a; 
			}

			if (max < epsilon) {
				break;
			}

			JacobiRotate(D, eigenVecs, p, q);
			iter++;
		}

		eigenVals[0] = D[0][0];
		eigenVals[1] = D[1][1];
		eigenVals[2] = D[2][2];
	}

	bool IsApprox(float a, float b);
	bool IsApprox(double a, double b);
	bool IsApprox(const glm::vec2& a, const glm::vec2& b);
	bool IsApprox(const glm::vec3& a, const glm::vec3& b);
	bool IsApprox(const glm::vec4& a, const glm::vec4& b);
	bool IsApprox(const glm::dvec2& a, const glm::dvec2& b);
	bool IsApprox(const glm::dvec3& a, const glm::dvec3& b);
	bool IsApprox(const glm::dvec4& a, const glm::dvec4& b);
	class Random {
	public:
		/// <summary>
		/// Generates a random float within the specified range.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>Randomly generated float. </returns>
		static float RandomFloat(float min, float max);

		/// <summary>
		/// Generates a random int within the specified range.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>Randomly generated int. </returns>
		static int RandomInt(int min, int max);

		/// <summary>
		/// Generates a ramdom boolean.
		/// </summary>
		/// <returns>Randomly generated boolean.</returns>
		static bool RandomBool();

		/// <summary>
		/// Generates a random 2-component vector, where each value falls into the specified range. Components of the vector are unique.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>A randomly generated vec2.</returns>
		static glm::vec2 RandomVec2(float min, float max);

		/// <summary>
		/// Generates a random 3-component vector, where each value falls into the specified range. Components of the vector are unique.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>A randomly generated vec3.</returns>
		static glm::vec3 RandomVec3(float min, float max);

		/// <summary>
		/// Generates a random 4-component vector, where each value falls into the specified range. Components of the vector are unique.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>A randomly generated vec4.</returns>
		static glm::vec4 RandomVec4(float min, float max);
	};
}

#endif // !MATH_H