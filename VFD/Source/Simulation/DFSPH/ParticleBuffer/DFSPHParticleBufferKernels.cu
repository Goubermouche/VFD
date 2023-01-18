#include "pch.h"
#include "DFSPHParticleBufferKernels.cuh"

__global__ void ConvertParticlesToBuffer(
	vfd::DFSPHParticle* source,
	vfd::DFSPHParticleSimple* destination,
	unsigned int particleCount
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= particleCount)
	{
		return;
	}

	// printf("{%.2f, %.2f, %.2f}\n", source[i].Position.x, source[i].Position.y, source[i].Position.z);
	destination[i].Position = source[i].Position;
}