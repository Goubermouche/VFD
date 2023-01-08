#ifndef MESH_DISTANCE_H
#define MESH_DISTANCE_H

#include "Renderer/Mesh/EdgeMesh.h"
#include "Core/Structures/Cache.h"

namespace vfd {
	class MeshDistance : public RefCounted
	{
	public:
		MeshDistance(const Ref<EdgeMesh>& mesh, bool preCalculateNormals = true);
		MeshDistance(const MeshDistance& other);

		double Distance(const glm::dvec3& x, glm::dvec3* closestPoint = nullptr, unsigned int* nearestFace = nullptr, Triangle* closestEntity = nullptr) const;
		void Callback(unsigned int nodeIndex, const glm::dvec3& point, double& distanceCandidate) const;
		bool Predicate(unsigned int nodeIndex, const Ref<MeshBoundingSphereHierarchy>& bsh, const glm::dvec3& point, double& distanceCandidate) const;

		double SignedDistance(const glm::dvec3& point) const;
		double SignedDistanceCached(const glm::dvec3& point) const;
	private:
		static double PointTriangleDistanceSquared(const glm::dvec3& point, const std::array<const glm::dvec3*, 3>& triangle, glm::dvec3* closestPoint = nullptr, Triangle* closestEntity = nullptr);

		glm::dvec3 CalculateVertexNormal(unsigned int vertex) const;
		glm::dvec3 CalculateEdgeNormal(const HalfEdge& halfEdge) const;
		glm::dvec3 CalculateFaceNormal(unsigned int face) const;
	private:
		Ref<EdgeMesh> m_Mesh;
		Ref<MeshBoundingSphereHierarchy> m_BSH;

		mutable std::vector<MeshBoundingSphereHierarchy::TraversalQueue> m_Queues;
		mutable std::vector<unsigned int> m_ClosestFace;
		mutable std::vector<Cache<glm::dvec3, double>> m_Cache;

		std::vector<glm::dvec3> m_FaceNormals;
		std::vector<glm::dvec3> m_VertexNormals;
		bool m_PreCalculatedNormals = true;
	};
}
#endif // !MESH_DISTANCE_H