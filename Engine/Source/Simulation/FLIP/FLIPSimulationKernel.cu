#ifndef FLIP_SIMULATION_KERNEL_CU
#define FLIP_SIMULATION_KERNEL_CU

#include "Compute/Utility/CUDA/cutil_math.h"
#include "Simulation/FLIP/FLIPSimulationParameters.cuh"

namespace fe {
	__constant__ FLIPSimulationParameters c_FLIPDescription;
	__device__ MACVelocityField c_MACTest;
	__device__ ParticleLevelSet d_ParticleLevelSet;
	__device__ MeshLevelSet d_LevelSet;
	// __device__ ViscositySolver d_VSolver;

	static __global__ void FLIPTestKernel() {
		printf("%.2f\n", c_MACTest.U.Get(0));
		printf("%.2f\n", c_MACTest.V(0));
		printf("%.2f\n", c_MACTest.W.Grid[0]);
	}
}

#endif // !FLIP_SIMULATION_KERNEL_CU