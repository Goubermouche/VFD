#include "pch.h"
#include "GPUDFSPHSimulator.h"
#include "Renderer/Renderer.h"
#include "Debug/SystemInfo.h"
#include "Utility/FileSystem.h"

namespace vfd
{
	GPUDFSPHSimulation::GPUDFSPHSimulation(GPUDFSPHSimulationDescription& desc)
	{
		if (SystemInfo::CUDADeviceMeetsRequirements() == false) {
			return;
		}

		//{
		//	RigidBodyDescription rigidbodyDesc;

		//	glm::mat4 transform(1.0f);
		//	//transform = glm::rotate(transform, 0.785398f, { 1.0f, 0.0f, 0.0f });
		//	//transform = glm::translate(transform, { 0.0f, -0.25f, 0.0f });
		//	//transform = glm::scale(transform, { 2.5f, 0.5f, 2.5f });

		//	rigidbodyDesc.Transform = transform;
		//	rigidbodyDesc.CollisionMapResolution = { 20, 20, 20 };
		//	// rigidbodyDesc.SourceMesh = "Resources/Models/Maxwell.obj";
		//	rigidbodyDesc.Inverted = false;
		//	rigidbodyDesc.Padding = 0.0f;

		//	m_RigidBodies.push_back(Ref<RigidBody>::Create(rigidbodyDesc));
		//	desc.BoundaryObjects.push_back(rigidbodyDesc);
		//}

		m_Implementation = Ref<DFSPHImplementation>::Create(desc);
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
		return m_Implementation->GetParticleRadius();
	}

	float GPUDFSPHSimulation::GetMaxVelocityMagnitude() const
	{
		return m_Implementation->GetMaxVelocityMagnitude();
	}

	float GPUDFSPHSimulation::GetCurrentTimeStepSize() const
	{
		return m_Implementation->GetTimeStepSize();
	}

	const ParticleSearch* GPUDFSPHSimulation::GetParticleSearch() const
	{
		return m_Implementation->GetParticleSearch();
	}

	const GPUDFSPHSimulationDescription& GPUDFSPHSimulation::GetDescription() const
	{
		return m_Implementation->GetDescription();
	}

	void GPUDFSPHSimulation::SetDescription(const GPUDFSPHSimulationDescription& desc)
	{
		m_Implementation->SetDescription(desc);
	}

	const DFSPHSimulationInfo& GPUDFSPHSimulation::GetInfo() const
	{
		return m_Implementation->GetInfo();
	}

	PrecomputedDFSPHCubicKernel& GPUDFSPHSimulation::GetKernel()
	{
		return m_Implementation->GetKernel();
	}

	const std::vector<Ref<RigidBody>>& GPUDFSPHSimulation::GetRigidBodies() const
	{
		return m_Implementation->GetRigidBodies();
	}

	void GPUDFSPHSimulation::Reset()
	{
		m_Implementation->Reset();
	}

	void GPUDFSPHSimulation::Simulate(std::vector<Ref<RigidBody>>& rigidBodies)
	{
		m_Implementation->Simulate(rigidBodies);
		paused = false;
	}

	void GPUDFSPHSimulation::OnUpdate()
	{
		if (paused) {
			return;
		}

		m_Implementation->OnUpdate();
	}
}
