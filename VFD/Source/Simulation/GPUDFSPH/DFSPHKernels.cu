#include "pch.h"
#include "DFSPHKernels.cuh"

__global__ void TestKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	info->Volume = 222.0f;

	// particles[i].Position.x += 0.1f;
}