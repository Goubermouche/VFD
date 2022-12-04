#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "DFSPHParticle.h"

__global__ void TestKernel(vfd::DFSPHParticle* particles);

#endif // !DFSPH_KERNELS_CUH