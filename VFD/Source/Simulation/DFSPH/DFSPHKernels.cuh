#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "Simulation/DFSPH/Structures/DFSPHParticle.h"
#include "Simulation/DFSPH/Structures/DFSPHParticleSimple.h"
#include "Simulation/DFSPH/Structures/DFSPHSimulationInfo.h"
#include "Simulation/DFSPH/RigidBody/RigidBodyDeviceData.cuh"
#include "Simulation/DFSPH/Kernel/DFSPHKernels.h"
#include "Simulation/DFSPH/ParticleSearch/NeighborSet.h"

#define MAX_CUDA_THREADS_PER_BLOCK 256

__global__ void ConvertParticlesToBuffer(
	vfd::DFSPHParticle* source,
	vfd::DFSPHParticleSimple* destination,
	vfd::DFSPHSimulationInfo* info
);

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
	vfd::RigidBodyDeviceData** rigidBodies
);

__global__ void ComputeDensityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputeDFSPHFactorKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputeDensityAdvectionKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void PressureSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationAndDivergenceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationAndVelocityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputeDensityChangeKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void DivergenceSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

__global__ void ComputePressureAccelerationAndFactorKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
);

// Viscosity solver
__global__ void ComputeViscosityPreconditionerKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::mat3x3* inverseDiagonal
);

__global__ void ComputeViscosityGradientKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::vec3* b,
	glm::vec3* g
);

__global__ void ComputeMatrixVecProdFunctionKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::vec3* rhs,
	glm::vec3* result
);

__global__ void ApplyViscosityForceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	glm::vec3* x
);

// Surface tension solver
__device__ bool ClassifyParticleConfigurable(
	const vfd::DFSPHSimulationInfo* info,
	float com,
	unsigned int non,
	float offset = 0.0f
);

__global__ void ComputeSurfaceTensionClassificationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet
);

__global__ void ComputeSurfaceTensionNormalsAndCurvatureKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet
);

__global__ void ComputeSurfaceTensionBlendingKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
);

#endif // !DFSPH_KERNELS_CUH