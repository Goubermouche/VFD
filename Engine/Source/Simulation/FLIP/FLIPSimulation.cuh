#ifndef FLIP_SIMULATION_CUH
#define FLIP_SIMULATION_CUH

#include "Simulation/FLIP/FLIPSimulationParameters.cuh"

namespace fe {
	extern "C" {
		void FLIPUploadSimulationParametersToSymbol(FLIPSimulationParameters& data);
		void FLIPUploadMACVelocitiesToSymbol(MACVelocityField& data);
		void FLIPUpdateFluidSDF();
	}
}
#endif // !FLIP_SIMULATION_CUH