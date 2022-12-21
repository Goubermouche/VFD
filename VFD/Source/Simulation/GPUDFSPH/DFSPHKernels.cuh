#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "RigidBody/RigidBodyDeviceData.cuh"
#include "Simulation/GPUDFSPH/Scalar/Vec3Vec8.h"
#include "NeigborhoodSearch/PointSetDeviceData.cuh"

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

__global__ void ComputeDensityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::PointSetDeviceData* pointSet,
	vfd::RigidBodyDeviceData* rigidBody
);

//__global__ void PreCalculateVolumeGradientWKernel(
//	vfd::DFSPHParticle* particles,
//	vfd::DFSPHSimulationInfo* info,
//	const unsigned int* precalculatedIndices,
//	vfd::vec3vec8* volumeGradient
//);

#endif // !DFSPH_KERNELS_CUH