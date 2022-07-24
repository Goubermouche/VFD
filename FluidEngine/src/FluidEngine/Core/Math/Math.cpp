#include "pch.h"
#include "Math.h"

namespace fe {
#define MAX_DEVIATION 0.0000001f
	bool DecomposeTransform(const glm::mat4& transform, glm::vec3& translation, glm::vec3& rotation, glm::vec3& scale)
	{
		using namespace glm;
		using T = float;

		mat4 LocalMatrix(transform);

		// Normalize the matrix
		if (epsilonEqual(LocalMatrix[3][3], static_cast<float>(0), epsilon<T>())) {
			return false;
		}

		// Perspective isolation
		if (
			epsilonNotEqual(LocalMatrix[0][3], static_cast<T>(0), epsilon<T>()) ||
			epsilonNotEqual(LocalMatrix[1][3], static_cast<T>(0), epsilon<T>()) ||
			epsilonNotEqual(LocalMatrix[2][3], static_cast<T>(0), epsilon<T>()))
		{
			// Clear the perspective partition
			LocalMatrix[0][3] = LocalMatrix[1][3] = LocalMatrix[2][3] = static_cast<T>(0);
			LocalMatrix[3][3] = static_cast<T>(1);
		}

		// Translation
		translation = vec3(LocalMatrix[3]);
		LocalMatrix[3] = vec4(0.0f, 0.0f, 0.0f, LocalMatrix[3].w);

		vec3 Row[3];

		// Scale
		for (length_t i = 0; i < 3; ++i) {
			for (length_t j = 0; j < 3; ++j) {
				Row[i][j] = LocalMatrix[i][j];
			}
		}

		// Compute X scale factor and normalize first row
		scale.x = length(Row[0]);
		Row[0] = detail::scale(Row[0], static_cast<T>(1));
		scale.y = length(Row[1]);
		Row[1] = detail::scale(Row[1], static_cast<T>(1));
		scale.z = length(Row[2]);
		Row[2] = detail::scale(Row[2], static_cast<T>(1));

		rotation.y = asin(-Row[0][2]);
		if (cos(rotation.y) != 0.0f) {
			rotation.x = atan2(Row[1][2], Row[2][2]);
			rotation.z = atan2(Row[0][1], Row[0][0]);
		}
		else {
			rotation.x = atan2(-Row[2][0], Row[1][1]);
			rotation.z = 0.0f;
		}

		return true;
	}

	bool IsApprox(float a, float b)
	{
		return (std::abs(a - b) <= MAX_DEVIATION);
	}

	bool IsApprox(const glm::vec3& a, const glm::vec3& b)
	{
		return IsApprox(a.x, b.x) && IsApprox(a.y, b.y) && IsApprox(a.z, b.z);
	}

	float Random::RandomFloat(float min, float max)
	{
		return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
	}

	int Random::RandomInt(int min, int max)
	{
		return min + static_cast <int> (rand()) / (static_cast <int> (RAND_MAX / (max - min)));
	}

	bool Random::RandomBool()
	{
		return RandomInt(0, 2);
	}

	glm::vec3 Random::RandomVec3(float min, float max)
	{
		return {
			RandomFloat(min, max),
			RandomFloat(min, max),
			RandomFloat(min, max),
		};
	}
}