#include "pch.h"
#include "StaticBoundarySimulator.h"

namespace fe {
	StaticRigidBody::StaticRigidBody(DFSPHSimulation* base, BoundaryData* data)
		: m_boundaryVolume(), m_boundaryXj() {
		m_base = base;
		m_map = nullptr;

		m_boundaryVolume.resize(1);
		m_boundaryXj.resize(1);

		m_boundaryVolume[0].resize(m_base->m_numParticles, 0.0f);
		m_boundaryXj[0].resize(m_base->m_numParticles, { 0.0f, 0.0f, 0.0f });

		std::string meshFileName = data->meshFile;
		TriangleMesh& geo = GetGeometry();
		geo.LoadOBJ(meshFileName, data->scale);
		geo.Translate(data->translation);

		glm::quat q = data->rotation;
		SetPosition(data->translation);
		setRotation(q);

		TriangleMesh& mesh = GetGeometry();
		m_base->InitVolumeMap(mesh.GetVertices(), mesh.GetTriangles(), data, false, false, this);
	}
}
