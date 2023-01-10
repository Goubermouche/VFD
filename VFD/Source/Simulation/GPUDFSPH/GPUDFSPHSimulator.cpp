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

	void GPUDFSPHSimulation::Simulate(const std::vector<Ref<RigidBody>>& rigidBodies)
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
