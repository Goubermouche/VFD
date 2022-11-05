#ifndef STATIC_BOUNDARY_SIMULATOR_H
#define STATIC_BOUNDARY_SIMULATOR_H

#include "DFSPHSimulation.h"
#include "Renderer/Mesh/TriangleMesh.h"
#include "Utility/SDF/SDF.h"

namespace fe {
	class DFSPHSimulation;
	struct BoundaryData;

	class StaticRigidBody {
	public:
		TriangleMesh m_geometry;
		glm::vec3 m_x;
		glm::quat m_q;
		SDF* m_map;
		std::vector<std::vector<float>> m_boundaryVolume;
		std::vector<std::vector<glm::vec3>> m_boundaryXj;
		DFSPHSimulation* m_base;

		StaticRigidBody(DFSPHSimulation* base, BoundaryData* data);
			
		TriangleMesh& GetGeometry() { return m_geometry; }
		void SetPosition(const glm::vec3& x) { m_x = x; }
		void setRotation(const glm::quat& q) { m_q = q; }

		inline void GetPointVelocity(const glm::vec3& x, glm::vec3& res) {
			res = { 0, 0, 0 };
		}

		inline glm::vec3& GetBoundaryXj(const unsigned int i) {
			return m_boundaryXj[0][i];
		}

		inline float& GetBoundaryVolume(const unsigned int i) {
			return m_boundaryVolume[0][i];
		}

		void SetMap(SDF* map) { m_map = map; }
	};
}
#endif // !STATIC_BOUNDARY_SIMULATOR_H