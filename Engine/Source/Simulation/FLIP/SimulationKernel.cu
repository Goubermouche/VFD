#ifndef FLIP_SIMULATION_KERNEL_CU
#define FLIP_SIMULATION_KERNEL_CU

#include "Compute/Utility/CUDA/cutil_math.h"
#include "SimulationData.cuh"

namespace fe {
	namespace flip {
		__constant__ SimulationData c_Description;
	}
}

#endif // !FLIP_SIMULATION_KERNEL_CU