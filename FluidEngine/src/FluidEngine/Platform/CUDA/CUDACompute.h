#ifndef CUDA_COMPUTE_H_
#define CUDA_COMPUTE_H_

#include "FluidEngine/Compute/ComputeAPI.h"
#include "FluidEngine/Platform/CUDA/Compute.cuh"

namespace fe::cuda {
	class CUDACompute: public ComputeAPI
	{
		// Inherited via ComputeAPI
		virtual void Init() override;
	private:
	};
}
#endif // !CUDA_COMPUTE_H_