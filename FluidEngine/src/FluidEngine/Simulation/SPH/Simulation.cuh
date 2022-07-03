#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include "SimulationKernel.cuh"
#include "cutil/inc/cutil_math.h"

namespace fe {
	extern "C" {
		int IDivUp(int a, int b);
		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount);
		void Integrate(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel, int particleCount);
		void CalculateHash(float4* pos, uint2* particleHash, int particleCount);
		void Reorder(uint2* particleHash, uint* cellStart, float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel, int particleCount, int cellCount);
		void CalculateDensity(float4* sortedPos, float* pressure, float* density, uint2* particleHash, uint* cellStart, int particleCount);
		void CalculateForce(float4* newPos, float4* newVel, float4* sortedPos, float4* sortedVel, float* pressure, float* density, uint2* particleHash, uint* cellStart, int particleCount);
		void Collide(float4* vboOldPos, float4* vboNewPos,
			float4* sortedPos, float4* sortedVel, float4* oldVel, float4* newVel,
			float* pressure, float* density,
			uint2* particleHash, uint* cellStart, uint numParticles, uint numCells);
	}
}

#endif // !SIMULATION_H_