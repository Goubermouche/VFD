#ifndef FLIP_SIMULATION_CUH
#define FLIP_SIMULATION_CUH

#include "Simulation/FLIP/FLIPSimulationData.cuh"

namespace fe {
	extern "C" {
		void FLIPUploadSimulationData(FLIPSimulationData& data);
		void FLIPUploadMACVelocity(MACVelocityField& mac);
		void FLIPUpdateFluidSDF();
	}
}
#endif // !FLIP_SIMULATION_CUH