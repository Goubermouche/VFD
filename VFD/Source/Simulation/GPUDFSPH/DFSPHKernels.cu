#include "pch.h"
#include "DFSPHKernels.cuh"

__global__ void TestKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	info->Volume += 0.1f;
}