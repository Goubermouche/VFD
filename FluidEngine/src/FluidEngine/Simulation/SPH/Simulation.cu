#include "Simulation.cuh"
#include <cuda_runtime.h>

namespace fe {
	extern "C" {
		void SetParameters(SimParams* params) {
			cudaMemcpyToSymbol(simulationParameters, params, sizeof(SimParams));
		}
	}
}