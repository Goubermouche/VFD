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

		float dx = 1.0f / std::max({ desc.Size.x, desc.Size.y, desc.Size.z });

		m_Parameters.TimeStep = desc.TimeStep / desc.SubStepCount;
		m_Parameters.SubStepCount = desc.SubStepCount;
		m_Parameters.Size = desc.Size;
		m_Parameters.DX = dx;
		m_Parameters.ParticleRadius = (float)(dx * 1.01f * std::sqrt(3.0f) / 2.0f);

		m_MACVelocity.Init(desc.Size.x, desc.Size.y, desc.Size.z, dx);
		m_ValidVelocities.Init(desc.Size.x, desc.Size.y, desc.Size.z);

		InitMemory();

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

	}

	void FLIPSimulation::InitMemory()
	{
		m_MACVelocityDevice = m_MACVelocity.UploadToDevice();

		FLIPUploadMACVelocitiesToSymbol(m_MACVelocityDevice);
		FLIPUploadSimulationParametersToSymbol(m_Parameters);

		m_Initialized = true;

		// TEMP test kernel
		FLIPUpdateFluidSDF();
	}

	void FLIPSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

		m_MACVelocityDevice.Free();
	}
}