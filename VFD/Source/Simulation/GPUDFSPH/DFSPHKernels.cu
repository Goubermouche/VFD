#include "pch.h"
#include "DFSPHKernels.cuh"

__device__ void WarpReduce(volatile float* data, unsigned int tid)
{
	data[tid] = min(data[tid], data[tid + 32]);
	data[tid] = min(data[tid], data[tid + 16]);
	data[tid] = min(data[tid], data[tid + 8]);
	data[tid] = min(data[tid], data[tid + 4]);
	data[tid] = min(data[tid], data[tid + 2]);
	data[tid] = min(data[tid], data[tid + 1]);
}

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

__global__ void MaxVelocityReductionKernel(vfd::DFSPHParticle* particles, float* output, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int tid = threadIdx.x;
	__shared__ float chunk[MAX_CUDA_THREADS_PER_BLOCK];

	if(i < info->ParticleCount)
	{
		// Calculate the velocity magnitude
		chunk[tid] = glm::length2(particles[i].Velocity + particles[i].Acceleration * info->TimeStepSize);
	}

	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
		__syncthreads();

		if (tid < stride)
		{
			chunk[tid] = max(chunk[tid], chunk[tid + stride]);
		}
	}

	if (tid < 32) {
		WarpReduce(chunk, tid);
	}

	if (tid == 0) {
		output[i] = chunk[0];
	}
}