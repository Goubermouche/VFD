#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "RigidBody/RigidBodyDeviceData.cuh"
#include "Kernel/DFSPHKernels.h"
#include "ParticleSearch/NeighborSet.h"

#define MAX_CUDA_THREADS_PER_BLOCK 256

__global__ void ClearAccelerationKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info
);

__global__ void ComputeVelocityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
);

__global__ void ComputePositionKernel(
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
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputeDFSPHFactorKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputeDensityAdvectionKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void PressureSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationAndDivergenceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationAndVelocityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputeDensityChangeKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void DivergenceSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationAndFactorKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

// Viscosity solver

__global__ void ComputeViscosityPreconditionerKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::mat3x3* inverseDiagonal
);

__global__ void ComputeViscosityGradientRHSKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	float* b,
	float* g
);

__global__ void ComputeMatrixVecProdFunctionKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	float* rhs,
	float* result
);

__global__ void SolvePreconditioner(
	vfd::DFSPHSimulationInfo* info,
	glm::mat3x3* inverseDiagonal,
	float* b,
	float* x
);

__global__ void ApplyViscosityForceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	float* x
);

#endif // !DFSPH_KERNELS_CUH