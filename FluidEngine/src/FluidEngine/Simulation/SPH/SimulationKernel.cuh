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
	__device__ float CalculateCellDensity(int3 gridPos, uint index, float4 pos, float4* oldPos, uint2* particleHash, uint* cellStart);
	__device__ float3 CalculateCellForce(int3 gridPos, uint index, float4 pos, float4 vel, float4* oldPos, float4* oldVel, float pres, float dens, float* pressure, float* density, uint2* particleHash, uint* cellStart);
	__device__ float3 CalculatePairForce(float4 RelPos, float4 RelVel, float p1_add_p2, float d1_mul_d2);

	__global__ void IntegrateKernel(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel);
	__global__ void CalculateHashKernel(float4* pos, uint2* particleHash);
	__global__ void ReorderKernel(uint2* particleHash, uint* cellStart, float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel);
	__global__ void CalculateDensityKernel(float4* oldPos, float* pressure, float* density, uint2* particleHash, uint* cellStart);
	__global__ void CalculateForceKernel(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel, float* pressure, float* density, uint2* particleHash, uint* cellStart);
}

#endif // !SIMULATION_KERNEL_CUH_