#include "pch.h"
#include "GPUDFSPHSimulator.h"
#include <Core/Structures/BoundingBox.h>
#include "Renderer/Renderer.h"

namespace vfd
{
	const float radius = 0.01f;

	GPUDFSPHSimulation::GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc)
		: m_Description(desc)
	{
	}

	void GPUDFSPHSimulation::OnRender()
	{
	
	}

	void GPUDFSPHSimulation::OnUpdate()
	{

	}
}