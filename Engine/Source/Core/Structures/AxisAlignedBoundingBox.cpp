#include "pch.h"
#include "AxisAlignedBoundingBox.h"

namespace fe {
	AxisAlignedBoundingBox::AxisAlignedBoundingBox(const glm::vec3& p, float w, float h, float d)
		: position(p), width(w), height(h), depth(d)
	{}

	AxisAlignedBoundingBox::AxisAlignedBoundingBox(const glm::vec3& p1, const glm::vec3& p2)
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

	AxisAlignedBoundingBox::AxisAlignedBoundingBox(const std::vector<glm::vec3>& points)
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

		for (uint32_t i = 0; i < points.size(); i++)
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

	AxisAlignedBoundingBox::AxisAlignedBoundingBox(const glm::ivec3& triangle, const std::vector<glm::vec3>& vertices)
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
}