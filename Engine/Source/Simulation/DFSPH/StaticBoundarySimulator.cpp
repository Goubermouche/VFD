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

			StaticRigidBody* rb = new StaticRigidBody();
			rb->m_isAnimated = m_base->boundaryModels[i]->isAnimated;
			TriangleMesh& geo = rb->GetGeometry();
			geo.LoadOBJ(meshFileName, m_base->boundaryModels[i]->scale);

			std::vector<glm::vec3> boundaryParticles;
			glm::quat q = m_base->boundaryModels[i]->rotation;
			rb->SetPosition0(m_base->boundaryModels[i]->translation);
			rb->SetPosition(m_base->boundaryModels[i]->translation);
			rb->setRotation0(q);
			rb->setRotation(q);

			// BoundaryHandlingMethods::Bender2019
			BoundaryModelBender2019* bm = new BoundaryModelBender2019(m_base);
			bm->InitModel(rb);
			m_base->m_boundaryModels = bm;
			TriangleMesh& mesh = rb->GetGeometry();
			m_base->InitVolumeMap(mesh.GetVertices(), mesh.GetTriangles(), m_base->boundaryModels[i], false, false, bm);
		}
	}

	void StaticBoundarySimulator::DefferedInit()
	{
		// performNeighborhoodSearchSort Z sort
		m_base->UpdateVMVelocity();
	}

	BoundaryModelBender2019::BoundaryModelBender2019(DFSPHSimulation* base)
		: m_boundaryVolume(), m_boundaryXj()
	{
		m_base = base;
		m_map = nullptr;
		m_maxDist = 0.0;
		m_maxVel = 0.0;
	}

	void BoundaryModelBender2019::InitModel(StaticRigidBody* rbo)
	{
		m_boundaryVolume.resize(1);
		m_boundaryXj.resize(1);

		m_boundaryVolume[0].resize(m_base->m_numParticles, 0.0);
		m_boundaryXj[0].resize(m_base->m_numParticles, { 0, 0, 0 });

		m_rigidBody = rbo;
	}
}
