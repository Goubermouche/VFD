#ifndef KERNEL_CU_
#define KERNEL_CU_

#include "Params.cuh"

namespace fe {
	__constant__ SimParams simulationParameters;
}

#endif // !KERNEL_CU_