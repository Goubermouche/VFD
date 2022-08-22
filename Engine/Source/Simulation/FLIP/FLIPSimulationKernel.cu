#ifndef FLIP_SIMULATION_KERNEL_CU
#define FLIP_SIMULATION_KERNEL_CU

#include "Compute/Utility/CUDA/cutil_math.h"
#include "Simulation/FLIP/FLIPSimulationData.cuh"

namespace fe {
	__constant__ FLIPSimulationData c_FLIPDescription;

	static __global__ void FLIPTestKernel() {
		// c_FLIPDescription.Test();
	}
}

#endif // !FLIP_SIMULATION_KERNEL_CU