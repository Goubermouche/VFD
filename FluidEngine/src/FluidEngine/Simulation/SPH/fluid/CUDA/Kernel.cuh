/*  SPH Kernel, Device code.  */
#pragma once

#include "cutil/inc/cutil_math.h"
#include "math_constants.h"
#include "Params.cuh"

#if USE_TEX
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint2, 1, cudaReadModeElementType> particleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;

texture<float, 1, cudaReadModeElementType> pressureTex;
texture<float, 1, cudaReadModeElementType> densityTex;

texture<float, 1, cudaReadModeElementType> dyeColorTex;
#endif

__constant__ SimParams par;

__device__ int3 calcGridPos(float4 p);	//  calculate position in uniform grid
__device__ uint calcGridHash(int3 gridPos);	//  calculate address in grid from position
__global__ void calcHashD(float4* pos, uint2* particleHash);
__global__ void reorderD(uint2* particleHash, uint* cellStart, // particle id sorted by hash
	float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel);
__device__ float3 collideSpheres(float4 posAB, float4 velAB, float radiusAB);
__device__ float3 collideSpheresR(float3 posAB, float3 relVel, float radiusAB);
__device__ float3 collideSpheresN(float3 posAB, float3 norm, float3 relVel, float radiusAB);
__device__ float compDensCell(int3 gridPos, uint index, float4 pos, float4* oldPos, uint2* particleHash, uint* cellStart);
__global__ void computeDensityD(float4* clr, float4* oldPos, float* pressure, float* density,
	uint2* particleHash, uint* cellStart);
__device__ float3 compForcePair(float4 RelPos, float4 RelVel, float p1_add_p2, float d1_mul_d2);
__device__ float3 compForceCell(int3 gridPos, uint index,
	float4 pos, float4 vel, float4* oldPos, float4* oldVel,
	float pres, float dens, float* pressure, float* density,
	uint2* particleHash, uint* cellStart);


__device__ void boundary(float3& pos, float3& vel);
__global__ void integrateD(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel);
__global__ void computeForceD(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel,
	float4* clr, float* pressure, float* density, float* dyeColor/**/, uint2* particleHash, uint* cellStart);
