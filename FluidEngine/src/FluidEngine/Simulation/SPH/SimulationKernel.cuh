#ifndef SIMULATION_KERNEL_CUH_
#define SIMULATION_KERNEL_CUH_

#include "cutil/inc/cutil_math.h"
#include "math_constants.h"
#include "Params.cuh"

// CHECK
#define __CUDACC__
#include "cuda_texture_types.h"

namespace fe {
	__constant__ SimulationParameters parameters;

	void SetParameters(SimulationParameters& params);

	__device__ void CalculateBoundary(float3& pos, float3& vel);
	__device__ int3 CalculateGridPosition(float4 p);

	__global__ void IntegrateKernel(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel);
	__global__ void CalculateHashKernel(float4* pos, uint2* particleHash);
	__global__ void ReorderKernel(uint2* particleHash, uint* cellStart, float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel);

}

#endif // !SIMULATION_KERNEL_CUH_