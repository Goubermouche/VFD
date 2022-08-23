#ifndef FLIP_SIMULATION_KERNEL_CU
#define FLIP_SIMULATION_KERNEL_CU

#include "Compute/Utility/CUDA/cutil_math.h"
#include "Simulation/FLIP/FLIPSimulationData.cuh"

namespace fe {
	__constant__ FLIPSimulationData c_FLIPDescription;
	__constant__ Array3D<int> c_stru;

	static __global__ void FLIPTestKernel() {
		// c_FLIPDescription.Test();
		// Array3D<int> arr1(-1, 1, 0);
		// MACVelocityField mac;
	}
}

#endif // !FLIP_SIMULATION_KERNEL_CU