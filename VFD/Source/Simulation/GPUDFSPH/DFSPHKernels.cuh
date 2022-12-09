#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"

#define MAX_CUDA_THREADS_PER_BLOCK 256

__device__ void WarpReduce(
	volatile float* data,
	unsigned int tid
);

__global__ void ClearAccelerationsKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info
);

__global__ void CalculateVelocitiesKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
);

__global__ void CalculatePositionsKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
);

__global__ void MaxVelocityReductionKernel(
	vfd::DFSPHParticle* particles,
	float* output,
	vfd::DFSPHSimulationInfo* info
);

#endif // !DFSPH_KERNELS_CUH