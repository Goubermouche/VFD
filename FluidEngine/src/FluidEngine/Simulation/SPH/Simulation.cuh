#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include "Kernel.cuh"
#include "cutil/inc/cutil_math.h"

namespace fe {
	extern "C" {
		void SetParameters(SimulationParameters* params);
		int IDivUp(int a, int b);
		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount);

		// Simulation
		void Integrate(uint vboOldPosition, uint vboNewPosition, float4* oldVelocity, float4* newVelocity, int particleCount);
	}
}

#endif // !SIMULATION_H_