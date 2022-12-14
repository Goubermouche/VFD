#ifndef DFSPH_KERNELS_CUH
#define DFSPH_KERNELS_CUH

#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "CollisionMap/RigidBodyData.h"

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

// TODO: move to the RigidBody class
__device__ unsigned int MultiToSingleIndex(
	vfd::RigidBodyData* rigidBody,
	const glm::uvec3& index
);

// TODO: move to the RigidBody class
__device__ glm::uvec3 SingleToMultiIndex(
	vfd::RigidBodyData* rigidBody,
	const unsigned int index
);

// TODO: move to the RigidBody class
__device__ vfd::BoundingBox<glm::dvec3> CalculateSubDomain(
	vfd::RigidBodyData* rigidBody,
	const glm::uvec3& index
);

// TODO: move to the RigidBody class
__device__ vfd::BoundingBox<glm::dvec3> CalculateSubDomain(
	vfd::RigidBodyData* rigidBody,
	unsigned int index
);

// TODO: move to the RigidBody class
__device__ void ShapeFunction(
	double(&res)[32],
	const glm::dvec3& xi,
	glm::dvec3(&gradient)[32]
);

// TODO: move to the RigidBody class
__device__ bool DetermineShapeFunctions(
	vfd::RigidBodyData* rigidBody,
	unsigned int fieldID, 
	const glm::dvec3& x,
	unsigned int(&cell)[32],
	glm::dvec3& c0, 
	double(&N)[32],
	glm::dvec3(&dN)[32]
);

// TODO: move to the RigidBody class
__device__ double Interpolate(
	vfd::RigidBodyData* rigidBody,
	unsigned int fieldID,
	const glm::dvec3& xi,
	unsigned int(&cell)[32],
	const glm::dvec3& c0,
	double(&N)[32]
);

// TODO: move to the RigidBody class
__device__ double Interpolate(
	vfd::RigidBodyData* rigidBody,
	unsigned int fieldID,
	const glm::dvec3& xi,
	unsigned int(&cell)[32],
	const glm::dvec3& c0,
	double(&N)[32],
	glm::dvec3& gradient,
	glm::dvec3(&dN)[32]
);

__global__ void ComputeVolumeAndBoundaryKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyData* rigidBody
);

#endif // !DFSPH_KERNELS_CUH