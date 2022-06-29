#ifndef SIMULATION_KERNEL_CUH_
#define SIMULATION_KERNEL_CUH_

#include "cutil/inc/cutil_math.h"
#include "math_constants.h"
#include "Params.cuh"

// CHECK
#define __CUDACC__
#include "cuda_texture_types.h"

namespace fe {
	__constant__ SimulationParameters parameters;

	void SetParameters(SimulationParameters& params);
}

#endif // !SIMULATION_KERNEL_CUH_