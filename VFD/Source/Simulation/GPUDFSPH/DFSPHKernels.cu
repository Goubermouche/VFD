#include "pch.h"
#include "DFSPHKernels.cuh"

__global__ void TestKernel(vfd::DFSPHParticle* particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// particles[i].Position.x += 0.1f;
}