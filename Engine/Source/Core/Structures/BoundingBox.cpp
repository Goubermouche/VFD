#include "pch.h"
#include "BoundingBox.h"

namespace fe {
	BoundingBox::BoundingBox(const std::vector<glm::dvec3>& vertices)
	{
		for (uint32_t i = 1; i < vertices.size(); ++i)
		{
			Extend(vertices[i]);
		}
	}

	BoundingBox::BoundingBox(glm::dvec3 min, glm::dvec3 max)
		: min(min), max(max)
	{ }

	void BoundingBox::SetEmpty()
	{
		constexpr double floatMin = std::numeric_limits<double>::min();
		constexpr double floatMax = std::numeric_limits<double>::max();
		min = { floatMax, floatMax, floatMax };
		max = { floatMin, floatMin, floatMin };
	}

	void BoundingBox::Extend(const glm::dvec3& vec)
	{
		min = glm::min(min, vec);
		max = glm::max(max, vec);
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