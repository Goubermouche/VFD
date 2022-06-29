#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include "SimulationKernel.cuh"
#include "cutil/inc/cutil_math.h"

namespace fe {
	extern "C" {
		int IDivUp(int a, int b);
		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount);
	}
}

#endif // !SIMULATION_H_