#include "pch.h"
#include "Math.h"

#define MAX_DEVIATION 0.0000001f

namespace fe {
	bool DecomposeTransform(const glm::mat4& transform, glm::vec3& Position, glm::vec3& Rotation, glm::vec3& Scale)
	{
		glm::mat4 localMatrix(transform);

		// Normalize the matrix
		if (glm::epsilonEqual(localMatrix[3][3], static_cast<float>(0), glm::epsilon<float>())) {
			return false;
		}

		// Perspective isolation
		if (
			glm::epsilonNotEqual(localMatrix[0][3], static_cast<float>(0), glm::epsilon<float>()) ||
			glm::epsilonNotEqual(localMatrix[1][3], static_cast<float>(0), glm::epsilon<float>()) ||
			glm::epsilonNotEqual(localMatrix[2][3], static_cast<float>(0), glm::epsilon<float>()))
		{
			// Clear the perspective partition
			localMatrix[0][3] = localMatrix[1][3] = localMatrix[2][3] = static_cast<float>(0);
			localMatrix[3][3] = static_cast<float>(1);
		}

		// Translation
		Position = glm::vec3(localMatrix[3]);
		localMatrix[3] = glm::vec4(0.0f, 0.0f, 0.0f, localMatrix[3].w);

		glm::vec3 row[3];

		// Scale
		for (glm::length_t i = 0; i < 3; ++i) {
			for (glm::length_t j = 0; j < 3; ++j) {
				row[i][j] = localMatrix[i][j];
			}
		}

		// Compute X scale factor and normalize first row
		Scale.x = length(row[0]);
		row[0] = glm::detail::Scale(row[0], static_cast<float>(1));
		Scale.y = length(row[1]);
		row[1] = glm::detail::Scale(row[1], static_cast<float>(1));
		Scale.z = length(row[2]);
		row[2] = glm::detail::Scale(row[2], static_cast<float>(1));

		Rotation.y = asin(-row[0][2]);
		if (cos(Rotation.y) != 0.0f) {
			Rotation.x = atan2(row[1][2], row[2][2]);
			Rotation.z = atan2(row[0][1], row[0][0]);
		}
		else {
			Rotation.x = atan2(-row[2][0], row[1][1]);
			Rotation.z = 0.0f;
		}

		return true;
	}

	bool IsApprox(const float a, const float b)
	{
		return (std::abs(a - b) <= MAX_DEVIATION);
	}

	bool IsApprox(const double a, const double b)
	{
		return (std::abs(a - b) <= MAX_DEVIATION);
	}

	bool IsApprox(const glm::vec2& a, const glm::vec2& b)
	{
		return IsApprox(a.x, b.x) && IsApprox(a.y, b.y);
	}

	bool IsApprox(const glm::vec3& a, const glm::vec3& b)
	{
		return IsApprox(a.x, b.x) && IsApprox(a.y, b.y) && IsApprox(a.z, b.z);
	}

	bool IsApprox(const glm::vec4& a, const glm::vec4& b)
	{
		return IsApprox(a.x, b.x) && IsApprox(a.y, b.y) && IsApprox(a.z, b.z) && IsApprox(a.w, b.w);
	}

	bool IsApprox(const glm::dvec2& a, const glm::dvec2& b)
	{
		return IsApprox(a.x, b.x) && IsApprox(a.y, b.y);
	}

	bool IsApprox(const glm::dvec3& a, const glm::dvec3& b)
	{
		return IsApprox(a.x, b.x) && IsApprox(a.y, b.y) && IsApprox(a.z, b.z);
	}

	bool IsApprox(const glm::dvec4& a, const glm::dvec4& b)
	{
		return IsApprox(a.x, b.x) && IsApprox(a.y, b.y) && IsApprox(a.z, b.z) && IsApprox(a.w, b.w);
	}

	float Random::RandomFloat(const float min, const float max)
	{
		return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
	}

	int Random::RandomInt(const int min, const int max)
	{
		return min + static_cast <int> (rand()) / (static_cast <int> (RAND_MAX / (max - min)));
	}

	bool Random::RandomBool()
	{
		return RandomInt(0, 2);
	}

	glm::vec3 Random::RandomVec3(const float min, const float max)
	{
		return {
			RandomFloat(min, max),
			RandomFloat(min, max),
			RandomFloat(min, max)
		};
	}

	glm::vec2 Random::RandomVec2(const float min, const float max)
	{
		return {
			RandomFloat(min, max),
			RandomFloat(min, max)
		};
	}

	glm::vec4 Random::RandomVec4(const float min, const float max)
	{
		return {
			RandomFloat(min, max),
			RandomFloat(min, max),
			RandomFloat(min, max),
			RandomFloat(min, max)
		};
	}
}