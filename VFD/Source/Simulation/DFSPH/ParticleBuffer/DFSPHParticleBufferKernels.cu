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

	destination[i].Position = source[i].Position;
}