#ifndef MATH_H_
#define MATH_H_

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cereal.hpp>

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
}

#endif // !MATH_H_