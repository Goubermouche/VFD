#include "pch.h"
#include "DFSPHSimulator.h"

#include "Renderer/Renderer.h"
#include "Debug/SystemInfo.h"

namespace vfd
{
	DFSPHSimulation::DFSPHSimulation(DFSPHSimulationDescription& desc)
	{
		if (SystemInfo::CUDADeviceMeetsRequirements() == false) {
			return;
		}

		m_Implementation = Ref<DFSPHImplementation>::Create(desc);
		m_Initialized = true;
	}

	DFSPHSimulation::~DFSPHSimulation()
	{
		if (m_Initialized == false) {
			return;
		}
	}

	const Ref<VertexArray>& DFSPHSimulation::GetVertexArray()
	{
		return m_Implementation->GetVertexArray();
	}

	void DFSPHSimulation::SetFluidObjects(const std::vector<Ref<FluidObject>>& fluidObjects)
	{
		m_Implementation->SetFluidObjects(fluidObjects);
	}

	void DFSPHSimulation::SetRigidBodies(const std::vector<Ref<RigidBody>>& rigidBodies)
	{
		m_Implementation->SetRigidBodies(rigidBodies);
	}

	DFSPHImplementation::SimulationState DFSPHSimulation::GetSimulationState() const
	{
		return m_Implementation->GetSimulationState();
	}

	unsigned int DFSPHSimulation::GetParticleCount()
	{
		return m_Implementation->GetParticleCount();
	}

	float DFSPHSimulation::GetParticleRadius() const
	{
		return m_Implementation->GetParticleRadius();
	}

	float DFSPHSimulation::GetMaxVelocityMagnitude() const
	{
		return m_Implementation->GetMaxVelocityMagnitude();
	}

	float DFSPHSimulation::GetCurrentTimeStepSize() const
	{
		return m_Implementation->GetCurrentTimeStepSize();
	}

	const ParticleSearch& DFSPHSimulation::GetParticleSearch() const
	{
		return m_Implementation->GetParticleSearch();
	}

	const DFSPHSimulationDescription& DFSPHSimulation::GetDescription() const
	{
		return m_Implementation->GetDescription();
	}

	void DFSPHSimulation::SetDescription(const DFSPHSimulationDescription& desc)
	{
		m_Implementation->SetDescription(desc);
	}

	const DFSPHSimulationInfo& DFSPHSimulation::GetInfo() const
	{
		return m_Implementation->GetInfo();
	}

	PrecomputedDFSPHCubicKernel& DFSPHSimulation::GetKernel()
	{
		return m_Implementation->GetKernel();
	}

	const std::vector<Ref<RigidBody>>& DFSPHSimulation::GetRigidBodies() const
	{
		return m_Implementation->GetRigidBodies();
	}

	unsigned int DFSPHSimulation::GetRigidBodyCount() const
	{
		return m_Implementation->GetRigidBodyCount();
	}

	const DFSPHDebugInfo& DFSPHSimulation::GetDebugInfo() const
	{
		return m_Implementation->GetDebugInfo();
	}

	Ref<DFSPHParticleBuffer> DFSPHSimulation::GetParticleFrameBuffer()
	{
		return m_Implementation->GetParticleFrameBuffer();
	}

	void DFSPHSimulation::Simulate()
	{
		std::thread simulationThread = std::thread(&DFSPHImplementation::Simulate, m_Implementation);
		simulationThread.detach();
	}	
}