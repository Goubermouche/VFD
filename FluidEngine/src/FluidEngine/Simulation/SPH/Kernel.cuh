#ifndef SPH_KERNEL_CUH_
#define SPH_KERNEL_CUH_

#include "Params.cuh"

namespace fe {
	__constant__ SimulationParameters parameters;

	__device__ void BoundaryKernel(float3& position, float3& velocity);

	__global__ void IntegrateKernel(float4* oldPosition, float4* newPosition, float4* oldVelocity, float4* newVelocity);
}

#endif // !SPH_KERNEL_CUH_
