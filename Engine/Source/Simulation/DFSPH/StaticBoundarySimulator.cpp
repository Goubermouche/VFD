#include "pch.h"
#include "StaticBoundarySimulator.h"
#include "Utility/SDF/MeshDistance.h"
#include "Core/Math/GaussQuadrature.h"

namespace fe {
	StaticRigidBody::StaticRigidBody(const StaticRigidBodyDescription& desc, DFSPHSimulation* base)
		: m_Description(desc), m_Base(base)
	{
		m_BoundaryVolume.resize(m_Base->GetParticleCount(), 0.0f);
		m_BoundaryXJ.resize(m_Base->GetParticleCount(), {0.0f, 0.0f, 0.0f});

		m_Geometry.LoadOBJ(m_Description.SourceMesh, m_Description.Scale);
		m_Geometry.Translate(m_Description.Position);

		m_Position = m_Description.Position;
		m_Rotation = m_Description.Rotation;

		// Init volume map
		const std::vector<glm::vec3>& x = m_Geometry.GetVertices();
		const std::vector<glm::ivec3>& faces = m_Geometry.GetTriangles();
		std::vector<glm::dvec3> doubleVec(x.size());

		const float supportRadius = m_Base->GetParticleSupportRadius();
		const float m_ParticleRadius = m_Base->GetParticleRadius();
		const float tolerance = m_Description.Padding;
		const float sign = m_Description.Inverted ? -1.0 : 1.0;

		for (unsigned int i = 0; i < x.size(); i++)
		{
			doubleVec[i] = glm::dvec3(x[i]);
		}

		EdgeMesh sdfMesh(doubleVec, faces);
		MeshDistance md(sdfMesh);
		BoundingBox domain(doubleVec);

		domain.max += (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);
		domain.min -= (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);

		m_CollisionMap = new SDF(domain, m_Description.CollisionMapResolution);
		m_CollisionMap->AddFunction([&md, &sign, &tolerance, &m_ParticleRadius](glm::dvec3 const& xi) {
			return sign * (md.SignedDistanceCached(xi) - tolerance);
		});

		BoundingBox intermediateDomain = BoundingBox(glm::dvec3(-supportRadius), glm::dvec3(supportRadius));

		m_CollisionMap->AddFunction([&](glm::dvec3 const& x)
		{
			auto distanceX = m_CollisionMap->Interpolate(0u, x);

			if (distanceX > 2.0 * supportRadius)
			{
				return 0.0;
			}

			auto integrand = [&](glm::dvec3 const& xi) -> double
			{
				if (glm::length2(xi) > supportRadius * supportRadius) {
					return 0.0;
				}

				const float distance = m_CollisionMap->Interpolate(0u, x + xi);

				if (distance <= 0.0) {
					return 1.0;
				}

				if (distance < supportRadius) {
					return static_cast<double>(CubicKernel::W(static_cast<float>(distance)) / CubicKernel::WZero());
				}

				return 0.0;
			};

			return  0.8 * GaussQuadrature::Integrate(integrand, intermediateDomain, 30);
		});
	}
}
