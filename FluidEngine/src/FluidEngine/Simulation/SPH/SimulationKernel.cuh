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

	__device__ void BoundaryKernel(float3& position, float3& velocity);
	__device__ int3 CalculateGridPosition(float4 position);
	__device__ float CalculateCellDensity(int3 gridPosition, uint index, float4 position, float4* oldPosition, uint2* particleHash, uint* cellStart);
	__device__ float3 CalculateCellForce(int3 gridPosition, uint index, float4 position, float4 velocity, float4* oldPosition, float4* oldVelocity, float pres, float dens, float* pressure, float* density, uint2* particleHash, uint* cellStart);
	__device__ float3 CalculatePairForce(float4 relativePosition, float4 relativeVelocity, float p1AddP2, float d1MulD2);

	__global__ void IntegrateKernel(float4* oldPosition, float4* newPosition, float4* oldVelocity, float4* newVelocity);
	__global__ void CalculateHashKernel(float4* position, uint2* particleHash);
	__global__ void ReorderKernel(uint2* particleHash, uint* cellStart, float4* oldPosition, float4* oldVelocity, float4* sortedPosition, float4* sortedVelocity);
	__global__ void CalculateDensityKernel(float4* oldPosition, float* pressure, float* density, uint2* particleHash, uint* cellStart);
	__global__ void CalculateForceKernel(float4* newPosition, float4* newVelocity, float4* oldPosition, float4* oldVelocity, float* pressure, float* density, uint2* particleHash, uint* cellStart);
}

#endif // !SIMULATION_KERNEL_CUH_