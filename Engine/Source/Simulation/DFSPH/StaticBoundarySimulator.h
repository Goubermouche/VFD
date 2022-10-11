#ifndef STATIC_BOUNDARY_SIMULATOR_H
#define STATIC_BOUNDARY_SIMULATOR_H

#include "DFSPHSimulation.h"
#include "Renderer/Mesh/TriangleMesh.h"
#include "Utility/SDF/SDF.h"

namespace fe {
	class DFSPHSimulation;
	class StaticRigidBody {
	public:
		bool m_isAnimated;
		glm::vec3 m_velocity;
		glm::vec3 m_angularVelocity;
		TriangleMesh m_geometry;
		glm::vec3 m_x0;
		glm::vec3 m_x;
		glm::quat m_q;
		glm::quat m_q0;

		StaticRigidBody() {
			m_isAnimated = false;
			m_velocity = { 0, 0, 0 };
			m_angularVelocity = { 0, 0, 0 };
		}

		TriangleMesh& GetGeometry() { return m_geometry; }
		void SetPosition0(glm::vec3 x) { m_x0 = x; }
		void SetPosition(const glm::vec3& x) { m_x = x; }
		void setRotation0(const glm::quat& q) { m_q0 = q; }
		void setRotation(const glm::quat& q) { m_q = q; }
	};

	class BoundaryModelBender2019 {
	public:
		BoundaryModelBender2019(DFSPHSimulation* base);
		void InitModel(StaticRigidBody* rbo);
		void SetMap(SDF* map) { m_map = map; }
		StaticRigidBody* GetRigidBody() { return m_rigidBody; }

		inline glm::vec3& GetBoundaryXj(const unsigned int i) {
			return m_boundaryXj[0][i];
		}

		inline float& GetBoundaryVolume(const unsigned int i) {
			return m_boundaryVolume[0][i];
		}

		inline void GetPointVelocity(const glm::vec3& x, glm::vec3& res) {
			res = { 0, 0, 0 };
		}

		SDF* m_map;
		std::vector<std::vector<float>> m_boundaryVolume;
		std::vector<std::vector<glm::vec3>> m_boundaryXj;
		float m_maxDist;
		float m_maxVel;
		DFSPHSimulation* m_base;
		StaticRigidBody* m_rigidBody;
	};

	class StaticBoundarySimulator
	{
	public:
		StaticBoundarySimulator(DFSPHSimulation* base);
		void InitBoundaryData();
		void DefferedInit();
	private:
		DFSPHSimulation* m_base;
	};
}
#endif // !STATIC_BOUNDARY_SIMULATOR_H