#include "pch.h"
#include "StaticBoundarySimulator.h"

namespace fe {
	StaticBoundarySimulator::StaticBoundarySimulator(DFSPHSimulation* base)
	{
		m_base = base;
	}

	void StaticBoundarySimulator::InitBoundaryData()
	{
		for (unsigned int i = 0; i < m_base->boundaryModels.size(); i++) {
			std::string meshFileName = m_base->boundaryModels[i]->meshFile;

			StaticRigidBody* rb = new StaticRigidBody(m_base);
			rb->m_isAnimated = m_base->boundaryModels[i]->isAnimated;
			TriangleMesh& geo = rb->GetGeometry();
			geo.LoadOBJ(meshFileName, m_base->boundaryModels[i]->scale);
			geo.Translate(m_base->boundaryModels[i]->translation);


			glm::quat q = m_base->boundaryModels[i]->rotation;
			rb->SetPosition0(m_base->boundaryModels[i]->translation);
			rb->SetPosition(m_base->boundaryModels[i]->translation);
			rb->setRotation0(q);
			rb->setRotation(q);

			rb->InitModel(rb);
			m_base->m_RigidBodies.push_back(rb);
			ERR("push")
			TriangleMesh& mesh = rb->GetGeometry();
			m_base->InitVolumeMap(mesh.GetVertices(), mesh.GetTriangles(), m_base->boundaryModels[i], false, false, rb);
		}
	}

	void StaticRigidBody::InitModel(StaticRigidBody* rbo)
	{
		m_boundaryVolume.resize(1);
		m_boundaryXj.resize(1);

		m_boundaryVolume[0].resize(m_base->m_numParticles, 0.0);
		m_boundaryXj[0].resize(m_base->m_numParticles, { 0.0, 0.0, 0.0 });
	}
}
