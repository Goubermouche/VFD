#include "pch.h"
#include "Compute.h"

#include "FluidEngine/Platform/CUDA/CUDACompute.h"

namespace fe {
	ComputeAPI* Compute::s_ComputeAPI = nullptr;

	void Compute::Init()
	{
		ASSERT(s_ComputeAPI, "compute API not set!");
		s_ComputeAPI->Init();
	}

	void Compute::SetAPI(ComputeAPIType api)
	{
		switch (api)
		{
		case fe::ComputeAPIType::None: s_ComputeAPI = nullptr; return;
		case fe::ComputeAPIType::CUDA: s_ComputeAPI = new cuda::CUDACompute; return;
		}

		ASSERT("unknown compute API!");
	}
}