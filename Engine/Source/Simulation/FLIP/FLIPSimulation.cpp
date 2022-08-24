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
		m_Data.DX = 1.0f / std::max({ desc.Size.x, desc.Size.y, desc.Size.z });

		//m_MACVelocity.SetDefault();
		//m_MACVelocity = MACVelocityField(desc.Size.x, desc.Size.y, desc.Size.z, m_Data.DX);

		m_MAC.Init(desc.Size.x, desc.Size.y, desc.Size.z, m_Data.DX);

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
		m_DeviceMAC = m_MAC;

		COMPUTE_SAFE(cudaMalloc((void**)&m_DeviceMAC.U.Grid, m_MAC.U.GetSize()));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeviceMAC.V.Grid, m_MAC.V.GetSize()));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeviceMAC.W.Grid, m_MAC.W.GetSize()));

		COMPUTE_SAFE(cudaMemcpy(m_DeviceMAC.U.Grid, m_MAC.U.Grid, m_MAC.U.GetSize(), cudaMemcpyHostToDevice));
		COMPUTE_SAFE(cudaMemcpy(m_DeviceMAC.V.Grid, m_MAC.V.Grid, m_MAC.V.GetSize(), cudaMemcpyHostToDevice));
		COMPUTE_SAFE(cudaMemcpy(m_DeviceMAC.W.Grid, m_MAC.W.Grid, m_MAC.W.GetSize(), cudaMemcpyHostToDevice));

		FLIPUploadMACVelocities(m_DeviceMAC);
		FLIPUploadSimulationData(m_Data);

		m_Initialized = true;

		FLIPUpdateFluidSDF();
	}

	void FLIPSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

		COMPUTE_SAFE(cudaFree(m_DeviceMAC.U.Grid));
		COMPUTE_SAFE(cudaFree(m_DeviceMAC.V.Grid));
		COMPUTE_SAFE(cudaFree(m_DeviceMAC.W.Grid));
	}
}