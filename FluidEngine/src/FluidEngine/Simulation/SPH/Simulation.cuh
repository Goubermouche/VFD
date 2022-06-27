#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include "Kernel.cu"

namespace fe {
	extern "C" {
		void SetParameters(SimParams* params);
	}
}

#endif // !SIMULATION_H_