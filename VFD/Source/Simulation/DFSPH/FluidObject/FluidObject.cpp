#include "pch.h"
#include "FluidObject.h"

namespace vfd
{
	FluidObject::FluidObject(const FluidObjectDescription& desc, const DFSPHSimulationInfo& info)
		: m_Description(desc)
	{
		const std::vector<glm::uvec3>& faces = m_Description.Mesh->GetTriangles();
		std::vector<glm::vec3> vertices = m_Description.Mesh->GetVertices();
		for (glm::vec3& v : vertices)
		{
			v = desc.Transform * glm::vec4(v, 1.0f);
		}

		const Ref<EdgeMesh> edgeMesh = Ref<EdgeMesh>::Create(vertices, faces);

		m_Positions = ParticleSampler::SampleMeshVolume(
			edgeMesh,
			info.ParticleRadius,
			desc.Resolution,
			desc.Inverted,
			desc.SampleMode
		);
	}

	const std::vector<glm::vec3>& FluidObject::GetPositions() const
	{
		return m_Positions;
	}

	unsigned int FluidObject::GetPositionCount() const
	{
		return m_Positions.size();
	}

	const glm::vec3& FluidObject::GetVelocity() const
	{
		return m_Description.Velocity;
	}
}