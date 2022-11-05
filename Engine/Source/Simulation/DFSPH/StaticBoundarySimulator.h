#ifndef STATIC_BOUNDARY_SIMULATOR_H
#define STATIC_BOUNDARY_SIMULATOR_H

#include "DFSPHSimulation.h"
#include "Renderer/Mesh/TriangleMesh.h"
#include "Utility/SDF/SDF.h"

namespace fe {
	class DFSPHSimulation;

	struct StaticRigidBodyDescription {
		glm::vec3 Position;
		glm::quat Rotation;
		glm::vec3 Scale;

		glm::ivec3 CollisionMapResolution;
		std::string SourceMesh;

		bool Inverted;
		float Padding;
	};

	class StaticRigidBody {
	public:
		StaticRigidBody(const StaticRigidBodyDescription& desc, DFSPHSimulation* base);
			
		TriangleMesh& GetGeometry() {
			return m_Geometry; 
		}

		inline glm::vec3& GetBoundaryXJ(const unsigned int i) {
			return m_BoundaryXJ[i];
		}

		inline float& GetBoundaryVolume(const unsigned int i) {
			return m_BoundaryVolume[i];
		}

		const glm::quat& GetRotation() const {
			return m_Rotation;
		}

		const glm::vec3& GetPosition() const {
			return m_Position;
		}

		SDF* GetCollisionMap() {
			return m_CollisionMap;
		}

		const StaticRigidBodyDescription& GetDescription() const {
			return m_Description;
		}
	private:
		StaticRigidBodyDescription m_Description;
		DFSPHSimulation* m_Base;
		TriangleMesh m_Geometry;
		SDF* m_CollisionMap;

		std::vector<float> m_BoundaryVolume;
		std::vector<glm::vec3> m_BoundaryXJ;

		// TODO: eventually replace these with a component accessor.
		glm::vec3 m_Position;
		glm::quat m_Rotation;
	};
}
#endif // !STATIC_BOUNDARY_SIMULATOR_H