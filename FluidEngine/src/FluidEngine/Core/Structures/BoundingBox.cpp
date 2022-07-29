#include "pch.h"
#include "BoundingBox.h"

namespace fe {
	BoundingBox::BoundingBox()
	{
		SetEmpty();
	}

	void BoundingBox::SetEmpty()
	{
		constexpr float floatMin = std::numeric_limits<float>::min();
		constexpr float floatMax = std::numeric_limits<float>::max();
		min = { floatMax ,floatMax ,floatMax };
		max = { floatMin ,floatMin ,floatMin };
	}

	void BoundingBox::Extend(const glm::vec3& vec)
	{
		min = glm::min(min, vec);
		max = glm::max(max, vec);
	}

	glm::vec3 BoundingBox::Diagonal()
	{
		return max - min;
	}

	glm::vec3 BoundingBox::Diagonal() const
	{
		return max - min;
	}

	bool BoundingBox::Contains(const glm::vec3& vec) const
	{
		return min.x <= vec.x && min.y <= vec.y && min.z <= vec.z && max.x >= vec.x && max.y >= vec.y && max.z >= vec.z;
	}

	BoundingBox BoundingBox::ComputeBoundingBox(const std::vector<glm::vec3>& vertices)
	{
		BoundingBox box;

		// Calculate bounding box	 
		box.min = vertices[0];
		box.max = box.min;
		box.SetEmpty();

		for (uint32_t i = 1; i < vertices.size(); ++i)
		{
			box.Extend(vertices[i]);
		}

		return box;
	}
}