#ifndef FLIP_SIMULATION_CUH
#define FLIP_SIMULATION_CUH

#include "SimulationData.cuh"

namespace fe {
	namespace flip {
		extern "C" {
			void UploadSimulationData(SimulationData& data);
		}
	}
}
#endif // !FLIP_SIMULATION_CUH