#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "RigidBody/RigidBodyDeviceData.cuh"

#define MAX_CUDA_THREADS_PER_BLOCK 256

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

__global__ void ComputeVolumeAndBoundaryKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData* rigidBody
);

#endif // !DFSPH_KERNELS_CUH