#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include "SimulationKernel.cuh"
#include "cutil/inc/cutil_math.h"

namespace fe {
	extern "C" {
		int IDivUp(int a, int b);
		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount);
		void Integrate(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel, int particleCount);
		void Hash(float4* pos, uint2* particleHash, int particleCount);
		void Reorder(uint2* particleHash, uint* cellStart, float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel, int particleCount, int cellCount);
	}
}

#endif // !SIMULATION_H_