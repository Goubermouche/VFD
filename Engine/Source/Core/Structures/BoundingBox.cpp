#include "pch.h"
#include "BoundingBox.h"

namespace fe {
	BoundingBox::BoundingBox(const std::vector<glm::vec3>& vertices)
	{
		for (uint32_t i = 1; i < vertices.size(); ++i)
		{
			Extend(vertices[i]);
		}
	}

	void BoundingBox::SetEmpty()
	{
		constexpr float floatMin = std::numeric_limits<float>::min();
		constexpr float floatMax = std::numeric_limits<float>::max();
		min = { floatMax, floatMax, floatMax };
		max = { floatMin, floatMin, floatMin };
	}

	void BoundingBox::Extend(const glm::vec3& vec)
	{
		min = glm::min((glm::vec3)min, vec);
		max = glm::max((glm::vec3)max, vec);
	}

	glm::dvec3 BoundingBox::Diagonal()
	{
		return max - min;
	}

	glm::dvec3 BoundingBox::Diagonal() const
	{
		return max - min;
	}

	bool BoundingBox::Contains(const glm::dvec3& vec) const
	{
		return min.x <= vec.x && min.y <= vec.y && min.z <= vec.z && max.x >= vec.x && max.y >= vec.y && max.z >= vec.z;
	}
}