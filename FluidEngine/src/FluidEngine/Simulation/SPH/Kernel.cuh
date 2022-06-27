#ifndef SPH_KERNEL_CUH_
#define SPH_KERNEL_CUH_

#include "Params.cuh"

namespace fe {
	__constant__ SimulationParameters parameters;

	__device__ void BoundaryKernel(float3& position, float3& velocity);
	__device__ int3 CalculateGridPosition(float4 position);

	__global__ void IntegrateKernel(float4* oldPosition, float4* newPosition, float4* oldVelocity, float4* newVelocity);
	__global__ void CalculateHashKernel(float4* position, uint2* particleHash);
}

#endif // !SPH_KERNEL_CUH_
