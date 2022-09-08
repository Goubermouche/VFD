#ifndef AXIS_ALIGNED_BOUNDING_BOX_H
#define AXIS_ALIGNED_BOUNDING_BOX_H

namespace fe {
	/// <summary>
	/// Axis aligned bounding box, defined by a position of it's corner and size. 
	/// </summary>
	class AABB
	{
	public:
		AABB() = default;
		AABB(const glm::vec3& p, float w, float h, float d);
		AABB(const glm::vec3& p1, const glm::vec3& p2);
		AABB(const std::vector<glm::vec3>& points);
		AABB(const glm::ivec3& triangle, const std::vector<glm::vec3>& vertices);

		glm::vec3 GetNearestPointInsideAABB(glm::vec3 p, double eps = 1e-6);

		void Expand(float v);
		
		bool IsPointInside(const glm::vec3& p);
		const glm::vec3 GetMinPoint();
		const glm::vec3 GetMaxPoint();
	public:
		glm::vec3 position = { 0.0f, 0.0f, 0.0f };
		float width = 0.0f;
		float height = 0.0f;
		float depth = 0.0f;
	};
}

#endif // !AXIS_ALIGNED_BOUNDING_BOX_H