#include "pch.h"
#include "GPUDFSPHSimulator.h"
#include <Core/Structures/BoundingBox.h>
#include "Renderer/Renderer.h"

namespace vfd
{
	GPUDFSPHSimulation::GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc)
		: m_Description(desc)
	{
		Ref<TriangleMesh> mesh = Ref<TriangleMesh>::Create("Resources/Models/Monkey.obj");
		m_SDF = Ref<GPUSDF>::Create(mesh, 100);

	}

	void GPUDFSPHSimulation::OnRender()
	{
		for (size_t i = 0; i < samples.size(); i++)
		{
			Renderer::DrawPoint(samples[i], { .5, .5, .5, 1.0f }, 0.1f);
		}
	}

	void GPUDFSPHSimulation::OnUpdate()
	{

	}
}