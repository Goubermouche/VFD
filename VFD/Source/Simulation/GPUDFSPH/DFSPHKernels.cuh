#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"

__global__ void TestKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info);

#endif // !DFSPH_KERNELS_CUH