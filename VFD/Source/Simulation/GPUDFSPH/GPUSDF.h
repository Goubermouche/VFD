#ifndef GPU_SDF_H
#define GPU_SDF_H

#include "Renderer/Mesh/TriangleMesh.h"
#include <Core/Structures/BoundingBox.h>

#include "Compute/Utility/Array3D.cuh"

namespace vfd {
	class GPUSDF : public RefCounted
	{
	public:
		GPUSDF(Ref<TriangleMesh>& mesh, float cellSize = 0.05f);

		// Sampling methods 
		float GetDistance(const glm::vec3& point);
		float GetDistanceTrilinear(const glm::vec3& point);
		float GetDistanceTricubic(const glm::vec3& point);

		const BoundingBox<glm::vec3>& GetDomain();
	private:
		static float PointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3);
		static float PointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2);
		static bool PointInTriangle2D(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3, double& a, double& b, double& c);
		static int Orientation(double x1, double y1, double x2, double y2, double& twice_signed_area);
		static void Sweep(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3D<float>& phi, Array3D<int>& closest_tri, const glm::vec3& origin, float dx, int di, int dj, int dk);
		static void CheckNeighbor(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3D<float>& phi, Array3D<int>& closest_tri, const glm::vec3& gx, int i0, int j0, int k0, int i1, int j1, int k1);
	private:
		int padding = 10.0f;
		float m_CellSize;
		float m_CellSizeInverse;
		unsigned int m_CellCount;
		glm::ivec3 m_Resolution;

		Array3D<float> m_PHI;
		BoundingBox<glm::vec3> m_Domain;
	};
}

#endif // !GPU_SDF_H