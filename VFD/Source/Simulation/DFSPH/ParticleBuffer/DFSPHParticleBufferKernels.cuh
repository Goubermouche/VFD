#ifndef DFSPH_PARTICLE_BUFFER_KERNELS_CUH
#define DFSPH_PARTICLE_BUFFER_KERNELS_CUH

#include "Simulation/DFSPH/Structures/DFSPHParticle.h"
#include "Simulation/DFSPH/Structures/DFSPHParticleSimple.h"

__global__ void ConvertParticlesToBuffer(
	vfd::DFSPHParticle* source,
	vfd::DFSPHParticleSimple* destination,
	unsigned int particleCount
);

#endif // !DFSPH_PARTICLE_BUFFER_KERNELS_CUH
