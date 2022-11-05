#include "pch.h"
#include "StaticBoundarySimulator.h"
#include "Utility/SDF/MeshDistance.h"
#include "Core/Math/GaussQuadrature.h"

namespace fe {
	StaticRigidBody::StaticRigidBody(const StaticRigidBodyDescription& desc, DFSPHSimulation* base)
		: m_Description(desc), m_Base(base)
	{
		m_boundaryVolume.resize(m_Base->m_numParticles, 0.0f);
		m_boundaryXj.resize(m_Base->m_numParticles, { 0.0f, 0.0f, 0.0f });

		TriangleMesh& geo = GetGeometry();
		geo.LoadOBJ(m_Description.meshFile, m_Description.scale);
		geo.Translate(m_Description.translation);

		SetPosition(m_Description.translation);
		setRotation(m_Description.rotation);

		TriangleMesh& mesh = GetGeometry();

		// Init volume map
		const auto& x = mesh.GetVertices();
		const auto& faces = mesh.GetTriangles();
		const float supportRadius = m_Base->m_supportRadius;
		std::vector<glm::dvec3> doubleVec(x.size());

		for (size_t i = 0; i < x.size(); i++)
		{
			doubleVec[i] = glm::dvec3(x[i].x, x[i].y, x[i].z);
		}

		EdgeMesh sdfMesh(doubleVec, faces);
		MeshDistance md(sdfMesh);
		BoundingBox domain;

		for (auto const& x_ : x)
		{
			domain.Extend(x_);
		}

		const float tolerance = m_Description.mapThickness;

		domain.max += (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);
		domain.min -= (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);


		m_map = new SDF(domain, m_Description.mapResolution);
		auto func = SDF::ContinuousFunction{};

		float sign = 1.0;
		if (m_Description.mapInvert) {
			sign = -1.0;
		}

		const float particleRadius = m_Base->particleRadius;
		// subtract 0.5 * particle radius to prevent penetration of particles and the boundary
		func = [&md, &sign, &tolerance, &particleRadius](glm::dvec3 const& xi) {return sign * (md.SignedDistanceCached(xi) - tolerance); };

		m_map->AddFunction(func);

		auto int_domain = BoundingBox(glm::dvec3(-supportRadius), glm::dvec3(supportRadius));
		float factor = 1.0;
		auto volume_func = [&](glm::dvec3 const& x)
		{
			auto dist_x = m_map->Interpolate(0u, x);

			if (dist_x > (1.0 + 1.0) * supportRadius)
			{
				return 0.0;
			}

			auto integrand = [&](glm::dvec3 const& xi) -> double
			{
				if (glm::length2(xi) > supportRadius * supportRadius)
					return 0.0;

				auto dist = m_map->Interpolate(0u, x + xi);

				if (dist <= 0.0)
					return 1.0;// -0.001 * dist / supportRadius;
				if (dist < 1.0 / factor * supportRadius)
					return static_cast<double>(CubicKernel::W(factor * static_cast<float>(dist)) / CubicKernel::WZero());
				return 0.0;
			};

			double res = 0.0;
			res = 0.8 * GaussQuadrature::Integrate(integrand, int_domain, 30);

			return res;
		};

		std::cout << "Generate volume map..." << std::endl;
		const bool no_reduction = true;
		m_map->AddFunction(volume_func, [&](glm::dvec3 const& x_)
			{
				if (no_reduction)
				{
					return true;
				}

				auto x = glm::max(x_, (glm::dvec3)glm::min((m_map->GetDomain().min), (m_map->GetDomain().max)));
				auto dist = m_map->Interpolate(0u, x);
				if (dist == std::numeric_limits<double>::max())
				{
					return false;
				}

				return fabs(dist) < 4.0 * supportRadius;
			});

	}
}
