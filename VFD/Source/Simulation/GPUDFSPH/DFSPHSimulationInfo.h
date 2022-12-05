#ifndef DFSPH_SIMULATION_INFO_H
#define DFSPH_SIMULATION_INFO_H

namespace vfd {
	struct DFSPHSimulationInfo {
		unsigned int ParticleCount;

		float SupportRadius;
		float TimeStepSize;
		float Volume;
		float Density0;
		float WZero;
	};
}

#endif // !DFSPH_SIMULATION_INFO_H