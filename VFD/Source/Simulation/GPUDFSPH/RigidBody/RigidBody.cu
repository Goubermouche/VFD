#include "pch.h"
#include "RigidBody.cuh"

#include "Compute/ComputeHelper.h"
#include "Core/Math/GaussQuadrature.h"

namespace vfd
{
	RigidBody::RigidBody(const RigidBodyDescription& desc)
		: m_Description(desc)
	{
		m_Mesh = Ref<TriangleMesh>::Create(desc.SourceMesh);

		m_Position = desc.Transform[3];

		glm::vec3 scale;
		for (int i = 0; i < 3; i++)
		{
			scale[i] = glm::length(glm::vec3(desc.Transform[i]));
		}

		m_Rotation = glm::mat3(
			glm::vec3(desc.Transform[0]) / scale[0],
			glm::vec3(desc.Transform[1]) / scale[1],
			glm::vec3(desc.Transform[2]) / scale[2]
		);

		// m_DensityMap = Ref<DensityMap>::Create("Resources/cache.cdm");
	}

	RigidBody::RigidBody(const RigidBodyDescription& desc, const DFSPHSimulationInfo& info, PrecomputedDFSPHCubicKernel& kernel)
		: m_Description(desc)
	{
		m_Position = desc.Transform[3];

		glm::vec3 scale;
		for (int i = 0; i < 3; i++)
		{
			scale[i] = glm::length(glm::vec3(desc.Transform[i]));
		}

		m_Rotation = glm::mat3(
			glm::vec3(desc.Transform[0]) / scale[0],
			glm::vec3(desc.Transform[1]) / scale[1],
			glm::vec3(desc.Transform[2]) / scale[2]
		);

		const std::vector<glm::vec3> boundaryXJ(info.ParticleCount, { 0.0f, 0.0f, 0.0f });
		const std::vector<float> boundaryVolume(info.ParticleCount, 0.0f);
		m_BoundaryXJ = boundaryXJ;
		m_BoundaryVolume = boundaryVolume;

		m_Mesh = Ref<TriangleMesh>::Create(desc.SourceMesh);

		// Initialize the volume map
		const std::vector<glm::vec3>& x = m_Mesh->GetVertices();
		const std::vector<glm::uvec3>& faces = m_Mesh->GetTriangles();
		std::vector<glm::dvec3> doubleVec(x.size());

		const float supportRadius = info.SupportRadius;
		const float particleRadius = info.ParticleRadius;
		const float tolerance = m_Description.Padding - particleRadius;
		const double sign = m_Description.Inverted ? -1.0 : 1.0;

		for (unsigned int i = 0; i < x.size(); i++)
		{
			doubleVec[i] = glm::dvec3(x[i]);
		}

		Ref<EdgeMesh> sdfMesh = Ref<EdgeMesh>::Create(doubleVec, faces);
		MeshDistance md(sdfMesh);
		BoundingBox<glm::dvec3> domain(doubleVec);

		domain.max += (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);
		domain.min -= (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);

		m_DensityMap = Ref<DensityMap>::Create(domain, m_Description.CollisionMapResolution);
		m_DensityMap->AddFunction([&](glm::dvec3 const& xi) -> double {
			return sign * (md.SignedDistanceCached(xi) - static_cast<double>(tolerance));
		});

		BoundingBox<glm::dvec3> intermediateDomain = BoundingBox<glm::dvec3>(glm::dvec3(-supportRadius), glm::dvec3(supportRadius));
		m_DensityMap->AddFunction([&](const glm::dvec3& x) -> double
		{
			const double d = m_DensityMap->Interpolate(0u, x);

			if (d > (1.0 + 1.0 / 5.0) * supportRadius)
			{
				return 0.0;
			}

			const auto integrand = [&](glm::dvec3 const& xi) -> double
			{
				if (glm::length2(xi) > supportRadius * supportRadius) {
					return 0.0;
				}

				const double dist = m_DensityMap->Interpolate(0u, x + xi);
				if (dist > 1.0 / 5.0 * supportRadius) {
					return 0.0;
				}

				return (1.0 - 5.0 * dist / supportRadius) * kernel.GetW(xi);
			};

			return 0.8 * GaussQuadrature::Integrate(integrand, intermediateDomain, 30);
		});
	}

	RigidBodyDeviceData* RigidBody::GetDeviceData()
	{
		auto* temp = new RigidBodyDeviceData();
		RigidBodyDeviceData* device;

		temp->Position = m_Position;
		temp->Rotation = m_Rotation;
		temp->BoundaryXJ = ComputeHelper::GetPointer(m_BoundaryXJ);
		temp->BoundaryVolume = ComputeHelper::GetPointer(m_BoundaryVolume);
		temp->Map = m_DensityMap->GetDeviceData();

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(RigidBodyDeviceData)))
		COMPUTE_SAFE(cudaMemcpy(device, temp, sizeof(RigidBodyDeviceData), cudaMemcpyHostToDevice))

		delete temp;
		return device;
	}

	const RigidBodyDescription& RigidBody::GetDescription()
	{
		return m_Description;
	}

	const Ref<TriangleMesh>& RigidBody::GetMesh()
	{
		return m_Mesh;
	}

	const glm::mat4& RigidBody::GetTransform()
	{
		return m_Description.Transform;
	}
}
