#ifndef MATH_H
#define MATH_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#define PI 3.141592654f
#define PI2 2.0f * PI

namespace fe {
	/// <summary>
	/// Decomposes a transform into its base components.
	/// </summary>
	/// <param name="transform">Transform matrix to decompose.</param>
	/// <param name="translation">Out translation.</param>
	/// <param name="rotation">Out rotation.</param>
	/// <param name="scale">Out scale</param>
	/// <returns>Whether the decomposition was successful</returns>
	bool DecomposeTransform(const glm::mat4& transform, glm::vec3& translation, glm::vec3& rotation, glm::vec3& scale);

	bool IsApprox(float a, float b);
	bool IsApprox(const glm::vec2& a, const glm::vec2& b);
	bool IsApprox(const glm::vec3& a, const glm::vec3& b);
	bool IsApprox(const glm::vec4& a, const glm::vec4& b);

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