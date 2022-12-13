#include "pch.h"
#include "DFSPHKernels.cuh"

__global__ void ClearAccelerationsKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Acceleration = info->Gravity;
}

__global__ void CalculateVelocitiesKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Velocity += info->TimeStepSize * particles[i].Acceleration;
}

__global__ void CalculatePositionsKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Position += info->TimeStepSize * particles[i].Velocity;
}

__global__ void ComputeVolumeAndBoundaryKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::RigidBodyData* rigidBodies)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	//printf("Domain min      (%.2f, %.2f, %.2f)\n", rigidBodies->Domain.min.x, rigidBodies->Domain.min.y, rigidBodies->Domain.min.z);
	//printf("Domain max      (%.2f, %.2f, %.2f)\n", rigidBodies->Domain.max.x, rigidBodies->Domain.max.y, rigidBodies->Domain.max.z);
	//printf("Resolution      (%u, %u, %u)\n", rigidBodies->Resolution.x, rigidBodies->Resolution.y, rigidBodies->Resolution.z);
	//printf("CellSize        (%.2f, %.2f, %.2f)\n", rigidBodies->CellSize.x, rigidBodies->CellSize.y, rigidBodies->CellSize.z);
	//printf("CellSizeInverse (%.2f, %.2f, %.2f)\n", rigidBodies->CellSizeInverse.x, rigidBodies->CellSizeInverse.y, rigidBodies->CellSizeInverse.z);
	//printf("CellCount       %llu\n", rigidBodies->CellCount);

	//printf("%f\n", rigidBodies->GetNode(0, 0));
	//printf("%u\n", rigidBodies->GetCellMap(0, 0));
	//printf("%u\n", rigidBodies->GetCell(0, 0, 0));
}
