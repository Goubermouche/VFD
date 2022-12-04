#include "pch.h"
#include "GPUDFSPHSimulator.h"
#include "Renderer/Renderer.h"

namespace vfd
{
	GPUDFSPHSimulation::GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc)
		: m_Description(desc)
	{
		m_Implementation = std::make_unique<DFSPHImplementation>();
	}

	const Ref<VertexArray>& GPUDFSPHSimulation::GetVertexArray()
	{
		return m_Implementation->GetVertexArray();
	}

	void GPUDFSPHSimulation::OnUpdate()
	{
		m_Implementation->OnUpdate();
	}
}