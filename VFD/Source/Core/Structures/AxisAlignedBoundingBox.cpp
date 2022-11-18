#include "pch.h"
#include "AxisAlignedBoundingBox.h"

namespace vfd {
	AABB::AABB(const glm::vec3& p, float w, float h, float d)
		: position(p), width(w), height(h), depth(d)
	{}

	AABB::AABB(const glm::vec3& p1, const glm::vec3& p2)
	{
		float minX = std::min(p1.x, p2.x);
		float minY = std::min(p1.y, p2.y);
		float minZ = std::min(p1.z, p2.z);

		float maxX = std::max(p1.x, p2.x);
		float maxY = std::max(p1.y, p2.y);
		float maxZ = std::max(p1.z, p2.z);

		position = { minX, minY, minZ };
		width = maxX - minX;
		height = maxY - minY;
		depth = maxZ - minZ;
	}

	AABB::AABB(const std::vector<glm::vec3>& points)
	{
		if (points.size() == 0) {
			return;
		}

		float minX = points[0].x;
		float minY = points[0].y;
		float minZ = points[0].z;

		float maxX = points[0].x;
		float maxY = points[0].y;
		float maxZ = points[0].z;

		for (unsigned int i = 0; i < points.size(); i++) {
			minX = std::min(points[i].x, minX);
			minY = std::min(points[i].y, minY);
			minZ = std::min(points[i].z, minZ);

			maxX = std::max(points[i].x, maxX);
			maxY = std::max(points[i].y, maxY);
			maxZ = std::max(points[i].z, maxZ);
		}

		float eps = 1e-9;
		position = glm::vec3(minX, minY, minZ);
		width = maxX - minX + eps;
		height = maxY - minY + eps;
		depth = maxZ - minZ + eps;
	}

	AABB::AABB(const glm::ivec3& triangle, const std::vector<glm::vec3>& vertices)
	{
		glm::vec3 points[3] = {
			vertices[triangle.x],
			vertices[triangle.y],
			vertices[triangle.z]
		};

		float minX = points[0].x;
		float minY = points[0].y;
		float minZ = points[0].z;

		float maxX = points[0].x;
		float maxY = points[0].y;
		float maxZ = points[0].z;

		for (uint32_t i = 0; i < 3; i++)
		{
			minX = std::min(points[i].x, minX);
			minY = std::min(points[i].y, minY);
			minZ = std::min(points[i].z, minZ);

			maxX = std::max(points[i].x, maxX);
			maxY = std::max(points[i].y, maxY);
			maxZ = std::max(points[i].z, maxZ);
		}

		float eps = 1e-9;
		position = { minX, minY, minZ };
		width = maxX - minX + eps;
		height = maxY - minY + eps;
		depth = maxZ - minZ + eps;
	}

	glm::vec3 AABB::GetNearestPointInsideAABB(glm::vec3 p, double eps)
	{
		if (IsPointInside(p)) {
			return p;
		}

		glm::vec3 min = GetMinPoint();
		glm::vec3 max = GetMaxPoint();

		p.x = fmax(p.x, min.x);
		p.y = fmax(p.y, min.y);
		p.z = fmax(p.z, min.z);

		p.x = fmin(p.x, max.x - (float)eps);
		p.y = fmin(p.y, max.y - (float)eps);
		p.z = fmin(p.z, max.z - (float)eps);

		return p;
	}

	void AABB::Expand(float v)
	{
		float h = 0.5f * v;
		position -= glm::vec3(h, h, h);
		width += v;
		height += v;
		depth += v;
	}

	bool AABB::IsPointInside(const glm::vec3& p)
	{
		return p.x >= position.x && p.y >= position.y && p.z >= position.z &&
			p.x < position.x + width && p.y < position.y + height && p.z < position.z + depth;
	}

	const glm::vec3 AABB::GetMinPoint()
	{
		return position;
	}

	const glm::vec3 AABB::GetMaxPoint()
	{
		return position + glm::vec3{ width, height, depth };
	}
}