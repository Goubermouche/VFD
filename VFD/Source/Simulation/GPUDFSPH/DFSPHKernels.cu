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

__global__ void ComputeVolumeAndBoundaryKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, vfd::RigidBodyDeviceData* rigidBody)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	glm::vec3& rigidBodyXJ = rigidBody->BoundaryXJ[i];
	float& rigidBodyVolume = rigidBody->BoundaryVolume[i];

	rigidBodyXJ = { 0.0f, 0.0f, 0.0f };
	rigidBodyVolume = 0.0f;

	glm::dvec3 normal;
	const glm::mat3& rotationMatrix = rigidBody->Rotation;
	const glm::dvec3 localPosition = glm::transpose(rotationMatrix) * glm::vec4(particlePosition, 0.0f);

	double dist = DBL_MAX;
	glm::dvec3 c0;
	unsigned int cell[32];
	double N[32];
	glm::dvec3 dN[32];

	if (rigidBody->Map->DetermineShapeFunction(0, localPosition, cell, c0, N, dN))
	{
		dist = rigidBody->Map->Interpolate(0, cell, c0, N, normal, dN);
	}

	if (dist > 0.0 && static_cast<float>(dist) < info->SupportRadius)
	{
		const double volume = rigidBody->Map->Interpolate(1, cell, N);
		if (volume > 0.0 && volume != DBL_MAX)
		{
			normal = rotationMatrix * normal;
			const double normalLength = glm::length(normal);

			if (normalLength > 1.0e-9)
			{
				rigidBodyVolume = static_cast<float>(volume);
				normal /= normalLength;
				const float d = glm::max((static_cast<float>(dist) + static_cast<float>(0.5) * info->ParticleRadius), info->ParticleDiameter);
				rigidBodyXJ = particlePosition - d * static_cast<glm::vec3>(normal);
			}
			else
			{
				rigidBodyVolume = 0.0f;
			}
		}
		else
		{
			rigidBodyVolume = 0.0f;
		}
	}
	else if (dist <= 0.0)
	{
		if (dist != DBL_MAX)
		{
			normal = rotationMatrix * normal;
			const double normalLength = glm::length(normal);

			if (normalLength > 1.0e-5)
			{
				normal /= normalLength;
				float delta = info->ParticleDiameter - static_cast<float>(dist);
				delta = glm::min(delta, static_cast<float>(0.1) * info->ParticleRadius);

				particles[i].Position = particlePosition + delta * static_cast<glm::vec3>(normal);
				particles[i].Velocity = { 0.0f, 0.0f, 0.0f };
			}
		}

		rigidBodyVolume = 0.0f;
	}
	else
	{
		rigidBodyVolume = 0.0f;
	}
}

__global__ void ComputeDensityKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, vfd::PointSetDeviceData* pointSet, vfd::RigidBodyDeviceData* rigidBody)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}
}

//__global__ void PreCalculateVolumeGradientWKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const unsigned int* precalculatedIndices, vfd::vec3vec8* volumeGradient)
//{
//	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (i >= info->ParticleCount)
//	{
//		return;
//	}
//}

// USE_AVX = 0
// PERF_OPTIMIZATION = 0


