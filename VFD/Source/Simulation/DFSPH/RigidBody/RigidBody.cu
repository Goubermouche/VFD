#include "pch.h"
#include "RigidBody.cuh"

#include "Compute/ComputeHelper.h"
#include "Core/Math/GaussQuadrature.h"
#include "Utility/SDF/MeshDistance.h"

namespace vfd
{
	RigidBody::RigidBody(const RigidBodyDescription& desc, const DFSPHSimulationInfo& info, PrecomputedDFSPHCubicKernel& kernel)
		: m_Description(desc)
	{
		const std::vector<glm::vec3> boundaryXJ(info.ParticleCount, { 0.0f, 0.0f, 0.0f });
		const std::vector<float> boundaryVolume(info.ParticleCount, 0.0f);
		m_BoundaryXJ = boundaryXJ;
		m_BoundaryVolume = boundaryVolume;

		// Initialize the volume map
		std::vector<glm::vec3> vertices = m_Description.Mesh->GetVertices();
		for(glm::vec3& v : vertices)
		{
			v = desc.Transform * glm::vec4(v, 1.0f);
		}

		const std::vector<glm::uvec3>& faces = m_Description.Mesh->GetTriangles();

		const float supportRadius = info.SupportRadius;
		const float particleRadius = info.ParticleRadius;
		const float tolerance = m_Description.Padding - particleRadius;
		const float sign = m_Description.Inverted ? -1.0f : 1.0f;

		Ref<EdgeMesh> sdfMesh = Ref<EdgeMesh>::Create(vertices, faces);
		MeshDistance md(sdfMesh);

		BoundingBox<glm::vec3> domain(vertices);
		domain.max += 8.0f * supportRadius + tolerance;
		domain.min -= 8.0f * supportRadius + tolerance;

		m_DensityMap = Ref<SDF>::Create(domain, m_Description.CollisionMapResolution);
		m_DensityMap->AddFunction([&](glm::vec3 const& xi) -> float {
			return sign * (md.SignedDistanceCached(xi) - static_cast<float>(tolerance));
		});

		BoundingBox<glm::vec3> intermediateDomain = BoundingBox<glm::vec3>(glm::vec3(-supportRadius), glm::vec3(supportRadius));
		m_DensityMap->AddFunction([&](const glm::vec3& x) -> float
		{
			const float distanceX = m_DensityMap->Interpolate(0u, x);
			if (distanceX > 2.0f * supportRadius)
			{
				return 0.0f;
			}

			const auto integrand = [&](glm::vec3 const& xi) -> float
			{
				if (glm::length2(xi) > supportRadius * supportRadius) {
					return 0.0f;
				}

				const float distance = m_DensityMap->Interpolate(0u, x + xi);
				if (distance <= 0.0f) {
					return 1.0f;
				}

				if (distance < supportRadius) {
					return kernel.GetW(distance) / kernel.GetWZero();
				}

				return 0.0f;
			};

			return 0.8f * GaussQuadrature::Integrate(integrand, intermediateDomain, 30);
		});
	}

	RigidBody::~RigidBody()
	{
		if(d_DeviceData != nullptr)
		{
			COMPUTE_SAFE(cudaFree(d_DeviceData));
		}
	}

	RigidBodyDeviceData* RigidBody::GetDeviceData()
	{
		if(d_DeviceData != nullptr)
		{
			return d_DeviceData;
		}

		auto* temp = new RigidBodyDeviceData();

		temp->BoundaryXJ = ComputeHelper::GetPointer(m_BoundaryXJ);
		temp->BoundaryVolume = ComputeHelper::GetPointer(m_BoundaryVolume);
		temp->Map = m_DensityMap->GetDeviceData();

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_DeviceData), sizeof(RigidBodyDeviceData)));
		COMPUTE_SAFE(cudaMemcpy(d_DeviceData, temp, sizeof(RigidBodyDeviceData), cudaMemcpyHostToDevice));

		delete temp;
		return d_DeviceData;
	}

	const RigidBodyDescription& RigidBody::GetDescription()
	{
		return m_Description;
	}

	const BoundingBox<glm::vec3>& RigidBody::GetBounds() const
	{
		return m_DensityMap->GetBounds();
	}
}
