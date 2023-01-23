#include "pch.h"
#include "DFSPHSimulator.h"

#include "Core/Application.h"
#include "Renderer/Renderer.h"
#include "Debug/SystemInfo.h"

namespace vfd
{
	DFSPHSimulation::DFSPHSimulation(const DFSPHSimulationDescription& desc)
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

	bool& DFSPHSimulation::GetRenderParticles()
	{
		return m_RenderParticles;
	}

	bool& DFSPHSimulation::GetRenderFlowLines()
	{
		return m_RenderFlowLines;
	}

	unsigned int DFSPHSimulation::GetFlowLineSampleCount() const
	{
		return m_FlowLineSampleCount;
	}

	const std::vector<unsigned int>& DFSPHSimulation::GetFlowLineIndices() const
	{
		return m_FlowLineIndices;
	}

	void DFSPHSimulation::RecomputeFlowLineIndices()
	{
		m_FlowLineIndices.resize(m_FlowLineSampleCount);

		const unsigned int max = m_Implementation->GetParticleCount();
		const float step = static_cast<float>(max) / static_cast<float>(m_FlowLineSampleCount);
		float position = 0.0f;

		for (unsigned int i = 0u; i < m_FlowLineSampleCount; i++)
		{
			m_FlowLineIndices[i] = static_cast<unsigned int>(position);
			position += step;
		}
	}

	void DFSPHSimulation::SetFlowLineCount(unsigned int count)
	{
		if(m_FlowLineSampleCount != count)
		{
			m_FlowLineSampleCount = std::min(count, m_Implementation->GetParticleCount());
			RecomputeFlowLineIndices();
		}
	}

	void DFSPHSimulation::Simulate()
	{
		Application::Get().GetThreadPool()->PushTask([&]
		{
			m_Implementation->Simulate();

			// Flow lines
			m_FlowLineSampleCount = glm::min(m_FlowLineSampleCount, m_Implementation->GetParticleCount());
			RecomputeFlowLineIndices();
		});
	}	
}