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

		m_MAC.Arr1.Init(10, 10, 10, 111.0f);
		m_MAC.Arr2.Init(10, 10, 10, 222.0f);
		m_MAC.Arr3.Init(10, 10, 10, 333.0f);

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

		COMPUTE_SAFE(cudaMalloc((void**)&m_DeviceMAC.Arr1.Grid, m_MAC.Arr1.ElementCount * sizeof(m_MAC.Arr1.Grid[0])));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeviceMAC.Arr2.Grid, m_MAC.Arr2.ElementCount * sizeof(m_MAC.Arr2.Grid[0])));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeviceMAC.Arr3.Grid, m_MAC.Arr3.ElementCount * sizeof(m_MAC.Arr3.Grid[0])));

		COMPUTE_SAFE(cudaMemcpy(m_DeviceMAC.Arr1.Grid, m_MAC.Arr1.Grid, m_MAC.Arr1.ElementCount * sizeof(m_MAC.Arr1.Grid[0]), cudaMemcpyHostToDevice));
		COMPUTE_SAFE(cudaMemcpy(m_DeviceMAC.Arr2.Grid, m_MAC.Arr2.Grid, m_MAC.Arr2.ElementCount * sizeof(m_MAC.Arr2.Grid[0]), cudaMemcpyHostToDevice));
		COMPUTE_SAFE(cudaMemcpy(m_DeviceMAC.Arr3.Grid, m_MAC.Arr3.Grid, m_MAC.Arr3.ElementCount * sizeof(m_MAC.Arr3.Grid[0]), cudaMemcpyHostToDevice));

		FLIPUploadMAC(m_DeviceMAC);
		FLIPUploadSimulationData(m_Data);

		m_Initialized = true;

		FLIPUpdateFluidSDF();
	}

	void FLIPSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

		COMPUTE_SAFE(cudaFree(m_DeviceMAC.Arr1.Grid));
		COMPUTE_SAFE(cudaFree(m_DeviceMAC.Arr2.Grid));
		COMPUTE_SAFE(cudaFree(m_DeviceMAC.Arr3.Grid));
	}
}