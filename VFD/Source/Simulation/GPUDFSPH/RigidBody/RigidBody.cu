#include "pch.h"
#include "RigidBody.cuh"

#include "Compute/ComputeHelper.h"
#include "Core/Math/GaussQuadrature.h"

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
		std::vector<glm::dvec3> verticesDouble(vertices.size());

		const float supportRadius = info.SupportRadius;
		const float particleRadius = info.ParticleRadius;
		const float tolerance = m_Description.Padding - particleRadius;
		const double sign = m_Description.Inverted ? -1.0 : 1.0;

		for (unsigned int i = 0; i < vertices.size(); i++)
		{
			verticesDouble[i] = glm::dvec3(vertices[i]);
		}

		Ref<EdgeMesh> sdfMesh = Ref<EdgeMesh>::Create(verticesDouble, faces);
		MeshDistance md(sdfMesh);
		BoundingBox<glm::dvec3> domain(verticesDouble);

		domain.max += (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);
		domain.min -= (8.0 * supportRadius + tolerance) * glm::dvec3(1.0);

		m_DensityMap = Ref<DensityMap>::Create(domain, m_Description.CollisionMapResolution);
		m_DensityMap->AddFunction([&](glm::dvec3 const& xi) -> double {
			return sign * (md.SignedDistanceCached(xi) - static_cast<double>(tolerance));
		});

		BoundingBox<glm::dvec3> intermediateDomain = BoundingBox<glm::dvec3>(glm::dvec3(-supportRadius), glm::dvec3(supportRadius));
		m_DensityMap->AddFunction([&](const glm::dvec3& x) -> double
		{
			const double distanceX = m_DensityMap->Interpolate(0u, x);

			if (distanceX > 2.0 * supportRadius)
			{
				return 0.0;
			}

			const auto integrand = [&](glm::dvec3 const& xi) -> double
			{
				if (glm::length2(xi) > supportRadius * supportRadius) {
					return 0.0;
				}

				const float distance = m_DensityMap->Interpolate(0u, x + xi);

				if (distance <= 0.0) {
					return 1.0;
				}

				if (distance < supportRadius) {
					return kernel.GetW(distance) / kernel.GetWZero();
				}

				return 0.0;
			};

			return 0.8 * GaussQuadrature::Integrate(integrand, intermediateDomain, 30);
		});
	}

	RigidBodyDeviceData* RigidBody::GetDeviceData()
	{
		auto* temp = new RigidBodyDeviceData();
		RigidBodyDeviceData* device;

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

	const BoundingBox<glm::dvec3>& RigidBody::GetBounds() const
	{
		return m_DensityMap->GetBounds();
	}

	//const Ref<TriangleMesh>& RigidBody::GetMesh()
	//{
	//	return m_Mesh;
	//}

	//const glm::mat4& RigidBody::GetTransform()
	//{
	//	return m_Description.Transform;
	//}
}
