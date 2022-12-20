#include "pch.h"
#include "GPUDFSPHSimulator.h"
#include "Renderer/Renderer.h"
#include "Debug/SystemInfo.h"
#include "Utility/FileSystem.h"

namespace vfd
{
	GPUDFSPHSimulation::GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc)
		: m_Description(desc)
	{
		if (SystemInfo::CUDADeviceMeetsRequirements() == false) {
			return;
		}

		m_RigidBodyMaterial = Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl"));
		m_RigidBodyMaterial->Set("color", { 0.4f, 0.4f, 0.4f, 1 });
		
		{
			RigidBodyDescription rigidbodyDesc;

			glm::mat4 transform(1.0f);
			transform = glm::rotate(transform, 0.785398f, { 1.0f, 0.0f, 0.0f });
			transform = glm::translate(transform, { 0.0f, -0.25f, 0.0f });
			transform = glm::scale(transform, { 2.5f, 0.5f, 2.5f });

			rigidbodyDesc.Transform = transform;
			rigidbodyDesc.CollisionMapResolution = { 10, 10, 10 };
			rigidbodyDesc.SourceMesh = "Resources/Models/Cube.obj";
			rigidbodyDesc.Inverted = false;
			rigidbodyDesc.Padding = 0.0f;

			m_RigidBodies.push_back(Ref<RigidBody>::Create(rigidbodyDesc));
		}

		m_Implementation = Ref<DFSPHImplementation>::Create(desc, m_RigidBodies);
		m_Initialized = true;
	}

	GPUDFSPHSimulation::~GPUDFSPHSimulation()
	{
		if (m_Initialized == false) {
			return;
		}
	}

	const Ref<VertexArray>& GPUDFSPHSimulation::GetVertexArray()
	{
		return m_Implementation->GetVertexArray();
	}

	unsigned int GPUDFSPHSimulation::GetParticleCount()
	{
		return m_Implementation->GetParticleCount();
	}

	float GPUDFSPHSimulation::GetParticleRadius() const
	{
		return m_Description.ParticleRadius;
	}

	Ref<Material>& GPUDFSPHSimulation::GetRigidBodyMaterial()
	{
		return m_RigidBodyMaterial;
	}

	float GPUDFSPHSimulation::GetMaxVelocityMagnitude() const
	{
		return m_Implementation->GetMaxVelocityMagnitude();
	}

	float GPUDFSPHSimulation::GetCurrentTimeStepSize() const
	{
		return m_Implementation->GetTimeStepSize();
	}

	std::vector<Ref<RigidBody>>& GPUDFSPHSimulation::GetRigidBodies()
	{
		return m_RigidBodies;
	}

	void GPUDFSPHSimulation::Reset()
	{
		m_Implementation->Reset();
	}

	void GPUDFSPHSimulation::OnUpdate()
	{
		if (paused) {
			return;
		}

		m_Implementation->OnUpdate();
	}
}
