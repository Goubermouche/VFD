#include "pch.h"
#include "RigidBody2.cuh"

#include "Compute/ComputeHelper.h"

namespace vfd
{
	RigidBody2::RigidBody2(const RigidBody2Description& desc)
		: m_Description(desc)
	{
		m_Mesh = Ref<TriangleMesh>::Create(desc.SourceMesh);

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

		m_DensityMap = DensityMap2("Resources/b.cdm");
	}

	RigidBody2DeviceData* RigidBody2::GetDeviceData(unsigned int particleCount)
	{
		// TEMP
		const std::vector<glm::vec3> boundaryXJ(particleCount);
		const std::vector<float> boundaryVolume(particleCount);
		m_BoundaryXJ = boundaryXJ;
		m_BoundaryVolume = boundaryVolume;

		auto* temp = new RigidBody2DeviceData();
		RigidBody2DeviceData* device;

		temp->Rotation = m_Rotation;
		temp->BoundaryXJ = ComputeHelper::GetPointer(m_BoundaryXJ);
		temp->BoundaryVolume = ComputeHelper::GetPointer(m_BoundaryVolume);
		temp->Map = m_DensityMap.GetDeviceData();

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(RigidBody2DeviceData)))
		COMPUTE_SAFE(cudaMemcpy(device, temp, sizeof(RigidBody2DeviceData), cudaMemcpyHostToDevice))

		delete temp;
		return device;
	}

}
