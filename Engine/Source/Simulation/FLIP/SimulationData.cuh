#ifndef FLIP_SIMULATION_DATA_CUH
#define FLIP_SIMULATION_DATA_CUH

#include "pch.h"

namespace fe {
	namespace flip {
		struct SimulationData
		{ 
			float TimeStep; // Time step / Sub step count

			uint32_t SubStepCount;
		};
	}
}

#endif // !FLIP_SIMULATION_DATA_CUH