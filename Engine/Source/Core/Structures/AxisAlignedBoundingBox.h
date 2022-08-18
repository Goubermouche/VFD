#ifndef AXIS_ALIGNED_BOUNDING_BOX_H
#define AXIS_ALIGNED_BOUNDING_BOX_H

namespace fe {
	class AxisAlignedBoundingBox
	{
	public:
		AxisAlignedBoundingBox() = default;
		AxisAlignedBoundingBox(const glm::vec3& p, float w, float h, float d);
		AxisAlignedBoundingBox(const glm::vec3& p1, const glm::vec3& p2);
		AxisAlignedBoundingBox(const std::vector<glm::vec3>& points);
		AxisAlignedBoundingBox(const glm::ivec3& triangle, const std::vector<glm::vec3>& vertices);

	public:
		glm::vec3 position = { 0.0f, 0.0f, 0.0f };
		float width = 0.0f;
		float height = 0.0f;
		float depth = 0.0f;
	};
}

#endif // !AXIS_ALIGNED_BOUNDING_BOX_H