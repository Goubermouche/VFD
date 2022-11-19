#ifndef MATH_H
#define MATH_H

#define PI 3.14159265358979323846
#define PI2 2.0 * PI
#define EPS 1.0e-5

namespace vfd {
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

		x = cross(vec, v);
		y = cross(vec, x);
		x = normalize(x);
		y = normalize(y);
	}

	static void JacobiRotate(glm::mat3x3& A, glm::mat3x3& R, const int p, const int q) {
		if (A[p][q] == 0.0f) {
			return;
		}

		const float d = (A[p][p] - A[q][q]) / (static_cast<float>(2.0) * A[p][q]);
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
				A[k][p] = A[p][k] = c * A[k][p] + s * A[k][q];
				A[k][q] = A[q][k] = -s * A[k][p] + c * A[k][q];
			}
		}

		for (k = 0; k < 3; k++) {
			R[k][p] = c * R[k][p] + s * R[k][q];
			R[k][q] = -s * R[k][p] + c * R[k][q];
		}
	}

	static void EigenDecomposition(const glm::mat3x3& A, glm::mat3x3& eigenVecs, glm::vec3& eigenVals) {
		constexpr int numJacobiIterations = 10;
		constexpr float epsilon = static_cast<float>(1e-15);

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
		static float Float(float min, float max);

		/// <summary>
		/// Generates a random int within the specified range.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>Randomly generated int. </returns>
		static int Int(int min, int max);

		/// <summary>
		/// Generates a ramdom boolean.
		/// </summary>
		/// <returns>Randomly generated boolean.</returns>
		static bool Bool();

		/// <summary>
		/// Generates a random 2-component vector, where each value falls into the specified range. Components of the vector are unique.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>A randomly generated vec2.</returns>
		static glm::vec2 Vec2(float min, float max);

		/// <summary>
		/// Generates a random 3-component vector, where each value falls into the specified range. Components of the vector are unique.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>A randomly generated vec3.</returns>
		static glm::vec3 Vec3(float min, float max);

		/// <summary>
		/// Generates a random 4-component vector, where each value falls into the specified range. Components of the vector are unique.
		/// </summary>
		/// <param name="min">Min value (inclusive).</param>
		/// <param name="max">Max value (exclusive).</param>
		/// <returns>A randomly generated vec4.</returns>
		static glm::vec4 Vec4(float min, float max);
	};
}

#endif // !MATH_H