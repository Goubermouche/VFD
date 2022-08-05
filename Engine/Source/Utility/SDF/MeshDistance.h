#ifndef MESH_DISTANCE_H
#define MESH_DISTANCE_H

#include "Renderer/Mesh/EdgeMesh.h"
#include "Core/Structures/Cache.h"

namespace fe {
	class MeshDistance : public RefCounted
	{
	public:
		MeshDistance(EdgeMesh const& mesh, bool preCalculateNormals = true);
		~MeshDistance() = default;

		float Distance(const glm::vec3& x, glm::vec3* closestPoint = nullptr, uint32_t* nearestFace = nullptr, Triangle* closestEntity = nullptr) const;
		void Callback(uint32_t nodeIndex, const glm::vec3& point, float& distanceCandidate) const;
		bool Predicate(uint32_t nodeIndex, const MeshBoundingSphereHierarchy& bsh, const glm::vec3& point, float& distanceCandidate) const;

		float SignedDistance(const glm::vec3& point) const;
		float SignedDistanceCached(const glm::vec3& point) const;
	private:
		static float PointTriangleDistanceSquared(const glm::vec3& point, const std::array<const glm::vec3*, 3>& triangle, glm::vec3* closestPoint = nullptr, Triangle* closestEntity = nullptr);

		glm::vec3 CalculateVertexNormal(uint32_t vertex) const;
		glm::vec3 CalculateEdgeNormal(const HalfEdge& halfEdge) const;
		glm::vec3 CalculateFaceNormal(uint32_t face) const;
	private:
		const EdgeMesh& m_Mesh;
		MeshBoundingSphereHierarchy m_BSH;

		mutable std::vector<MeshBoundingSphereHierarchy::TraversalQueue> m_Queues;
		mutable std::vector<uint32_t> m_ClosestFace;
		mutable std::vector<Cache<glm::vec3, float>> m_Cache;

		std::vector<glm::vec3> m_FaceNormals;
		std::vector<glm::vec3> m_VertexNormals;
		bool m_PreCalculatedNormals;
	};
}
#endif // !MESH_DISTANCE_H