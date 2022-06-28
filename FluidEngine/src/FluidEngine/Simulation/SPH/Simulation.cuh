#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include "SimulationKernel.cuh"
#include "cutil/inc/cutil_math.h"

namespace fe {
	extern "C" {
		void SetParameters(SimulationParameters* params);
		int IDivUp(int a, int b);
		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount);
		void CalculateHash(uint vboPosition, uint2* particleHash, int particleCount);
		void Integrate(uint vboOldPosition, uint vboNewPosition, float4* oldVelocity, float4* newVelocity, int particleCount);
		void Reorder(uint vboOldPosition, float4* oldVelocity, float4* sortedPosition, float4* sortedVelocity, uint2* particleHash, uint* cellStart, uint particleCount, uint cellCount);
		void Collide(uint vboOldPosition, uint vboNewPosition, float4* sortedPosition, float4* sortedVelocity, float4* oldVelocity, float4* newVelocity, float* pressure, float* density, /*dye*/ uint2* particleHash, uint* cellStart, uint particleCount, uint cellCount);
	}
}

#endif // !SIMULATION_H_