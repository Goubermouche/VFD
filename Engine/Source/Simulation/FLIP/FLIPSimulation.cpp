#include "pch.h"
#include "FLIPSimulation.h"

#include "Simulation/FLIP/FLIPSimulation.cuh"

#include <Glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	FLIPSimulation::FLIPSimulation(const FLIPSimulationDescription& desc)
		: m_Description(desc)
	{
		if (GPUCompute::GetInitState() == false) {
			// The GPU compute context failed to initialize. Return.
			ERR("Simulation stopped (GPU compute context failed to initialize)")
			return;
		}

		m_Data.TimeStep = desc.TimeStep / desc.SubStepCount;
		m_Data.SubStepCount = desc.SubStepCount;
		m_Data.Size = desc.Size;
		m_Data.DX = 1.0f / std::max({ desc.Size.x, desc.Size.y, desc.Size.z});

		m_MACVelocity = MACVelocityField(desc.Size.x, desc.Size.y, desc.Size.z, m_Data.DX);
		m_ValidVelocities = ValidVelocityComponentGrid(desc.Size.x, desc.Size.y, desc.Size.z);

		InitMemory();

		FLIPUploadSimulationData(m_Data);

		LOG("simulation initialized", "FLIP");
	}

	FLIPSimulation::~FLIPSimulation()
	{
	}

	void FLIPSimulation::OnUpdate()
	{
		if (m_Initialized == false || paused) {
			return;
		}

		FLIPUpdateFluidSDF();
	}

	void FLIPSimulation::InitMemory()
	{


		m_Initialized = true;
	}

	void FLIPSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}
	}
}