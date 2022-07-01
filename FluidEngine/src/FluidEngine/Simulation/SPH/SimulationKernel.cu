#include "SimulationKernel.cuh"
#include "cutil/inc/cutil.h"
#include <iostream>

namespace fe {
	void SetParameters(SimulationParameters& params)
	{
		printf("parameters set!\n");
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(parameters, &params, sizeof(SimulationParameters)));
	}

	__device__ void CalculateBoundary(float3& pos, float3& vel)
	{
		//  world box
		float3 wmin = parameters.worldMin, wmax = parameters.worldMax;

		float b = parameters.distBndSoft, stiff = parameters.bndStiff, damp = parameters.bndDamp, damp2 = parameters.bndDampC;
		float accBnd, diff;  float3 norm;

#define  EPS	0.00001f	// epsilon for collision detection
#define  addB()  accBnd = stiff * diff - damp * dot(norm, vel);  vel += accBnd * norm * parameters.timeStep;	// box,pump, soft
#define  addC()  accBnd = stiff * diff - damp2* dot(norm, vel);  vel += accBnd * norm * parameters.timeStep;	// cyl,sphr

		//----------------  Box
		if (true)
		{
			if (!false) {
				if (true || false) { diff = b - pos.z + wmin.z;	if (diff > EPS) { norm = make_float3(0, 0, 1);  addC(); } }
				if (!false) { diff = b + pos.z - wmax.z;	if (diff > EPS) { norm = make_float3(0, 0, -1);  addC(); } }
			}
			if (!false && !false) {
				diff = b - pos.x + wmin.x;	if (diff > EPS) { norm = make_float3(1, 0, 0);  addB(); }
				diff = b + pos.x - wmax.x;	if (diff > EPS) { norm = make_float3(-1, 0, 0);  addB(); }
			}

			if (!false) {
				diff = b - pos.y + wmin.y;	if (diff > EPS) { norm = make_float3(0, 1, 0);  addB(); }
				diff = b + pos.y - wmax.y;	if (diff > EPS) { norm = make_float3(0, -1, 0);  addB(); }
			}
		}
		else	//  Sphere
			/*if (t == BND_SPHERE)*/ {
			float len = length(pos);	diff = b + len + wmin.y;
			if (diff > EPS) { norm = make_float3(-pos.x / len, -pos.y / len, -pos.z / len);  addC(); }
		}

		//  Cylinder Y|
		if (false || false) {
			float len = length(make_float2(pos.x, pos.z));		diff = b + len - wmax.x;
			if (diff > EPS) { norm = make_float3(-pos.x / len, 0, -pos.z / len);  addC(); }
		}

		//  Cylinder Z--
		if (false) {
			float len = length(make_float2(pos.x, pos.y));		diff = b + len + wmin.y;
			if (diff > EPS) { norm = make_float3(-pos.x / len, -pos.y / len, 0);  addC(); }
		}

		//  Wrap, Cycle  Z--
		//if (!false && !true) {
		//	float dr = 1.f * parameters.particleR;/*parameters.rDexit*/
		//	if (false &&
		//		vel.z > parameters.rvec && pos.z > wmax.z - b - dr) {
		//		pos.z -= wmax.z - wmin.z - 2 * b - dr;
		//	}
		//	else
		//		if (vel.z < -parameters.rVexit && pos.z < wmin.z + b + dr) { pos.z += wmax.z - wmin.z - 2 * b - dr; }
		//}


		///  Pump  ~~~~~~~~~~~~~~~~~~~~~~~~~~
	}

	__device__ int3 CalculateGridPosition(float4 p)
	{
		int3 gridPos;
		float3 gp = (make_float3(p) - parameters.worldMin) / parameters.cellSize;
		gridPos.x = floor(gp.x);	//gridPos.x = max(0, min(gridPos.x, par.gridSize.x-1));	// not needed
		gridPos.y = floor(gp.y);	//gridPos.y = max(0, min(gridPos.y, par.gridSize.y-1));  //(clamping to edges)
		gridPos.z = floor(gp.z);	//gridPos.z = max(0, min(gridPos.z, par.gridSize.z-1));
		return gridPos;
	}

	__global__ void IntegrateKernel(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		float4 pos4 = oldPos[index];	float4 vel4 = oldVel[index];
		float3 pos = make_float3(pos4);	float3 vel = make_float3(vel4);

		CalculateBoundary(pos, vel);

		///  Euler integration  -------------------------------
		vel += parameters.gravity * parameters.timeStep;	// v(t) = a(t) dt
		vel *= parameters.globalDamping;	// = 1
		pos += vel * parameters.timeStep;			// p(t+1) = p(t) + v(t) dt

		//----------------  Hard boundary
		float b = parameters.distBndHard;
		float3 wmin = parameters.worldMin, wmax = parameters.worldMax;
		if (pos.x > wmax.x - b)   pos.x = wmax.x - b;
		if (pos.x < wmin.x + b)   pos.x = wmin.x + b;
		if (pos.y > wmax.y - b)   pos.y = wmax.y - b;
		if (pos.y < wmin.y + b)   pos.y = wmin.y + b;
		if (pos.z > wmax.z - b)   pos.z = wmax.z - b;
		if (pos.z < wmin.z + b)   pos.z = wmin.z + b;

		// store new position and velocity
		newPos[index] = make_float4(1);
		newVel[index] = make_float4(1);
	}

	__device__ uint CalculateGridHash(int3 gridPos)
	{
		return __mul24(gridPos.z, parameters.gridSize_yx)
			+ __mul24(gridPos.y, parameters.gridSize.x) + gridPos.x;
	}

	__global__ void CalculateHashKernel(float4* pos, uint2* particleHash)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		float4 p = pos[index];

		// get address in grid
		int3 gridPos = CalculateGridPosition(p);
		uint gridHash = CalculateGridHash(gridPos);

		// store grid hash and particle index
		// particleHash[index] = make_uint2(gridHash, index);
	}

	__global__ void ReorderKernel(uint2* particleHash, uint* cellStart, float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		uint2 sortedData = particleHash[index];

		// Load hash data into shared memory so that we can look 
		// at neighboring particle's hash value without loading
		// two hash values per thread

		__shared__ uint sharedHash[257];
		sharedHash[threadIdx.x + 1] = sortedData.x;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			volatile uint2 prevData = particleHash[index - 1];
			sharedHash[0] = prevData.x;
		}

		__syncthreads();

		if (index == 0 || sortedData.x != sharedHash[threadIdx.x])
			cellStart[sortedData.x] = index;

		// Now use the sorted index to reorder the pos and vel data
		sortedPos[index] = oldPos[sortedData.y];
		sortedVel[index] = oldVel[sortedData.y];
	}
}