#include "SimulationKernel.cuh"
#include "cutil/inc/cutil.h"
#include <iostream>

namespace fe {
	void SetParameters(SimulationParameters& params)
	{
		printf("parameters set!\n");
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(parameters, &params, sizeof(SimulationParameters)));
	}
}