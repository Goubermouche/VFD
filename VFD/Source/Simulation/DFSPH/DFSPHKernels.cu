#include "pch.h"
#include "DFSPHKernels.cuh"

#include "Simulation/DFSPH/HaltonVec323.cuh"

__global__ void ClearAccelerationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Acceleration = info->Gravity;
}

__global__ void ComputeVelocityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Velocity += info->TimeStepSize * particles[i].Acceleration;
}

__global__ void ComputePositionKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Position += info->TimeStepSize * particles[i].Velocity;
}

__global__ void ComputeVolumeAndBoundaryKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData** rigidBodies
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePosition = particles[i].Position;

	for(unsigned int j = 0u; j < info->RigidBodyCount; j++)
	{
		vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
		glm::vec3& boundaryPosition = rigidBody->BoundaryXJ[i] = { 0.0f, 0.0f, 0.0f };
		float& boundaryVolume = rigidBody->BoundaryVolume[i] = 0.0f;

		glm::vec3 normal;
		float distance = FLT_MAX;
		glm::vec3 interpolationVector;
		unsigned int cell[32];
		float shapeFunction[32];
		glm::vec3 shapeFunctionDerivative[32];

		if (rigidBody->Map->DetermineShapeFunction(0, particlePosition, cell, interpolationVector, shapeFunction, shapeFunctionDerivative))
		{
			distance = rigidBody->Map->Interpolate(0, cell, interpolationVector, shapeFunction, normal, shapeFunctionDerivative);
		}

		if (distance > 0.0f && distance < info->SupportRadius)
		{
			const float volume = rigidBody->Map->Interpolate(1, cell, shapeFunction);
			if (volume > 0.0f && volume != FLT_MAX)
			{
				boundaryVolume = static_cast<float>(volume);
				const float normalLength = glm::length(normal);

				if (normalLength > 1.0e-9f)
				{
					normal /= normalLength;
					const float particleDistance = glm::max(distance + 0.5f * info->ParticleRadius, info->ParticleDiameter);
					boundaryPosition = particlePosition - particleDistance * normal;
				}
				else
				{
					boundaryVolume = 0.0f;
				}
			}
			else
			{
				boundaryVolume = 0.0f;
			}
		}
		else if (distance <= 0.0f)
		{
			if (distance != FLT_MAX)
			{
				const float normalLength = glm::length(normal);

				if (normalLength > 1.0e-5f)
				{
					normal /= normalLength;
					float delta = info->ParticleDiameter - distance;
					delta = glm::min(delta, 0.1f * info->ParticleRadius);

					particlePosition += delta * normal;
					particles[i].Velocity = { 0.0f, 0.0f, 0.0f };
				}
			}

			boundaryVolume = 0.0f;
		}
		else
		{
			boundaryVolume = 0.0f;
		}
	}
}

__global__ void ComputeDensityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet, 
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	float& particleDensity = particles[i].Density;
	particleDensity = info->Volume * kernel->GetWZero();

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++) {
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		particleDensity += info->Volume * kernel->GetW(particlePosition - particles[neighborIndex].Position);
	}

	for(unsigned int j = 0u; j < info->RigidBodyCount; j++)
	{
		vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f) {
			particleDensity += boundaryVolume * kernel->GetW(particlePosition - rigidBody->BoundaryXJ[i]);
		}
	}

	particleDensity *= info->Density0;
}

__global__ void ComputeDFSPHFactorKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies, 
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	glm::vec3 gradientPI = { 0.0f, 0.0f, 0.0f };
	float sumGradientPK = 0.0f;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
		sumGradientPK += glm::dot(gradientPJ, gradientPJ);
		gradientPI -= gradientPJ;
	}

	for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
	{
		vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f)
		{
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);
			gradientPI -= gradientPJ;
		}
	}

	sumGradientPK += glm::dot(gradientPI, gradientPI);
	particles[i].Factor = sumGradientPK > EPS ? particles[i].Factor = 1.0f / sumGradientPK : 0.0f;
}

__global__ void ComputeDensityAdvectionKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = particles[i].Velocity;
	float& particleDensityAdvection = particles[i].DensityAdvection;
	float delta = 0.0f;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		delta += glm::dot(particleVelocity - particles[neighborIndex].Velocity, kernel->GetGradientW(particlePosition - particles[neighborIndex].Position));
	}

	delta *= info->Volume;

	for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
	{
		vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f)
		{
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			delta += boundaryVolume * glm::dot(particleVelocity, kernel->GetGradientW(particlePosition - neighborPosition));
		}
	}

	particleDensityAdvection = particles[i].Density / info->Density0 + info->TimeStepSize * delta;
	particles[i].Factor *= info->TimeStepSize2Inverse;
	const float si = 1.0f - particleDensityAdvection;
	const float residuum = glm::min(si, 0.0f);
	particles[i].PressureRho2 = -residuum * particles[i].Factor;
}

__device__ float ComputeDensityPressureForce(
	const unsigned int i,
	const vfd::DFSPHParticle* particles,
	const vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleAcceleration = particles[i].PressureAcceleration;
	float densityPressureForce = 0.0f;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		densityPressureForce += glm::dot(particleAcceleration - particles[neighborIndex].PressureAcceleration, kernel->GetGradientW(particlePosition - neighborParticlePosition));
	}

	densityPressureForce *= info->Volume;

	for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
	{
		const vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f)
		{
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			densityPressureForce += boundaryVolume * glm::dot(particleAcceleration, kernel->GetGradientW(particlePosition - neighborPosition));
		}
	}

	return densityPressureForce;
}

__global__ void PressureSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet, 
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const float& particleDensityAdvection = particles[i].DensityAdvection;

	float densityPressureForce = ComputeDensityPressureForce(i, particles, info, pointSet, rigidBodies, kernel);
	densityPressureForce *= info->TimeStepSize2;

	float& residuum = particles[i].PressureResiduum;
	residuum = glm::min(1.0f - particleDensityAdvection - densityPressureForce, 0.0f);
	particles[i].PressureRho2 -= residuum * particles[i].Factor;
}

__global__ void ComputePressureAccelerationKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet, 
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const float particlePressureRho = particles[i].PressureRho2;
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const float pressureSum = particlePressureRho + particles[neighborIndex].PressureRho2;
		
		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
		{
			vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
			const float boundaryVolume = rigidBody->BoundaryVolume[i];

			if (boundaryVolume > 0.0f) {
				const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
				const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

				particlePressureAcceleration += particlePressureRho * gradientPJ;
			}
		}
	}
}

__global__ void ComputePressureAccelerationAndDivergenceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const float particlePressureRho = particles[i].PressureRho2V;
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const float pressureSum = particlePressureRho + particles[neighborIndex].PressureRho2V;

		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
		{
			vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
			const float boundaryVolume = rigidBody->BoundaryVolume[i];

			if (boundaryVolume > 0.0f) {
				const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
				const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

				particlePressureAcceleration += particlePressureRho * gradientPJ;
			}
		}
	}
}

__global__ void ComputePressureAccelerationAndVelocityKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const float particlePressureRho = particles[i].PressureRho2;
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		const float pressureSum = particlePressureRho + particles[neighborIndex].PressureRho2;

		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - neighborParticlePosition);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
		{
			vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
			const float boundaryVolume = rigidBody->BoundaryVolume[i];

			if (boundaryVolume > 0.0f) {
				const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
				const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

				particlePressureAcceleration += particlePressureRho * gradientPJ;
			}
		}
	}

	particles[i].Velocity += info->TimeStepSize * particlePressureAcceleration;
}

__global__ void ComputeDensityChangeKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	float& particleDensityAdvection = particles[i].DensityAdvection;
	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = particles[i].Velocity;

	particleDensityAdvection = 0.0f;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		particleDensityAdvection += glm::dot(particleVelocity - particles[neighborIndex].Velocity, kernel->GetGradientW(particlePosition - neighborParticlePosition));
	}

	particleDensityAdvection *= info->Volume;

	for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
	{
		vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f) {
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			particleDensityAdvection += boundaryVolume * glm::dot(particleVelocity, kernel->GetGradientW(particlePosition - neighborPosition));
		}
	}
	
	particleDensityAdvection = pointSet->GetNeighborCount(i) < 20u ? 0.0f : glm::max(particleDensityAdvection, 0.0f);

	float& factor = particles[i].Factor;
	factor *= info->TimeStepSizeInverse;
	particles[i].PressureRho2V = particleDensityAdvection * factor;
}

__global__ void DivergenceSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	float densityPressureForce = ComputeDensityPressureForce(i, particles, info, pointSet, rigidBodies, kernel);
	densityPressureForce *= info->TimeStepSize;

	float& residuum = particles[i].PressureResiduum;
	residuum = pointSet->GetNeighborCount(i) < 20 ? 0.0f : glm::min(-particles[i].DensityAdvection - densityPressureForce, 0.0f);

	particles[i].PressureRho2V -= residuum * particles[i].Factor;
}

__global__ void ComputePressureAccelerationAndFactorKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const glm::vec3& particlePosition = particles[i].Position;
	const float particlePressureRho = particles[i].PressureRho2V;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const float pressureSum = particlePressureRho + particles[neighborIndex].PressureRho2V;

		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
		{
			vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
			const float boundaryVolume = rigidBody->BoundaryVolume[i];

			if (boundaryVolume > 0.0f) {
				const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
				const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

				particlePressureAcceleration += particlePressureRho * gradientPJ;
			}
		}
	}

	particles[i].Velocity += info->TimeStepSize * particlePressureAcceleration;
	particles[i].Factor *= info->TimeStepSize;
}

__global__ void ComputeViscosityPreconditionerKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::mat3x3* inverseDiagonal
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::mat3x3 result = glm::mat3x3();
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);

		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		const float neighborParticleDensity = particles[neighborIndex].Density;

		const glm::vec3 positionDirection = particlePosition - neighborParticlePosition;
		const glm::vec3 gradientW = kernel->GetGradientW(positionDirection);

		result += 10.0f * info->DynamicViscosity * (info->ParticleMass / neighborParticleDensity) / (glm::length2(positionDirection) + 0.01f * info->SupportRadius2) * glm::outerProduct(positionDirection, gradientW);
	}

	if(info->DynamicBoundaryViscosity != 0.0f)
	{
		for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
		{
			vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
			const float boundaryVolume = rigidBody->BoundaryVolume[i];

			if (boundaryVolume > 0.0f) {
				const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
				const glm::vec3 neighborDirection = particlePosition - neighborPosition;
				glm::vec3 normal = -neighborDirection;
				const float normalLength = glm::length(normal);

				if (normalLength > 0.0001f)
				{
					normal /= normalLength;

					glm::vec3 t1;
					glm::vec3 t2;
					vfd::GetOrthogonalVectors(normal, t1, t2);

					const glm::vec3 position1 = neighborPosition - t1 * info->TangentialDistance;
					const glm::vec3 position2 = neighborPosition + t1 * info->TangentialDistance;
					const glm::vec3 position3 = neighborPosition - t2 * info->TangentialDistance;
					const glm::vec3 position4 = neighborPosition + t2 * info->TangentialDistance;

					const glm::vec3 positionDirection1 = particlePosition - position1;
					const glm::vec3	positionDirection2 = particlePosition - position2;
					const glm::vec3	positionDirection3 = particlePosition - position3;
					const glm::vec3	positionDirection4 = particlePosition - position4;

					const glm::vec3 gradientW1 = kernel->GetGradientW(positionDirection1);
					const glm::vec3 gradientW2 = kernel->GetGradientW(positionDirection2);
					const glm::vec3 gradientW3 = kernel->GetGradientW(positionDirection3);
					const glm::vec3 gradientW4 = kernel->GetGradientW(positionDirection4);

					const float volume = 0.25f * boundaryVolume;

					result += 10.0f * info->DynamicBoundaryViscosity * volume / (glm::length2(positionDirection1) + 0.01f * info->SupportRadius2) * glm::outerProduct(positionDirection1, gradientW1);
					result += 10.0f * info->DynamicBoundaryViscosity * volume / (glm::length2(positionDirection2) + 0.01f * info->SupportRadius2) * glm::outerProduct(positionDirection1, gradientW2);
					result += 10.0f * info->DynamicBoundaryViscosity * volume / (glm::length2(positionDirection3) + 0.01f * info->SupportRadius2) * glm::outerProduct(positionDirection1, gradientW3);
					result += 10.0f * info->DynamicBoundaryViscosity * volume / (glm::length2(positionDirection4) + 0.01f * info->SupportRadius2) * glm::outerProduct(positionDirection1, gradientW4);
				}
			}
		}
	}

	inverseDiagonal[i] = glm::inverse(glm::identity<glm::mat3x3>() - (info->TimeStepSize / particles[i].Density) * result);
}

__global__ void ComputeViscosityGradientKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::vec3* b,
	glm::vec3* g
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = particles[i].Velocity;
	glm::vec3 velocity = { 0.0f, 0.0f, 0.0f };

	if(info->DynamicBoundaryViscosity != 0.0f)
	{
		for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
		{
			vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
			const float boundaryVolume = rigidBody->BoundaryVolume[i];

			if (boundaryVolume > 0.0f) {
				const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
				const glm::vec3 positionDirection = particlePosition - neighborPosition;
				glm::vec3 normal = -positionDirection;
				const float normalLength = glm::length(normal);

				if (normalLength > 0.0001f)
				{
					normal /= normalLength;

					glm::vec3 t1;
					glm::vec3 t2;
					vfd::GetOrthogonalVectors(normal, t1, t2);

					const glm::vec3 position1 = neighborPosition - t1 * info->TangentialDistance;
					const glm::vec3 position2 = neighborPosition + t1 * info->TangentialDistance;
					const glm::vec3 position3 = neighborPosition - t2 * info->TangentialDistance;
					const glm::vec3 position4 = neighborPosition + t2 * info->TangentialDistance;

					const glm::vec3 positionDirection1 = particlePosition - position1;
					const glm::vec3	positionDirection2 = particlePosition - position2;
					const glm::vec3	positionDirection3 = particlePosition - position3;
					const glm::vec3	positionDirection4 = particlePosition - position4;

					const glm::vec3	gradientW1 = kernel->GetGradientW(positionDirection1);
					const glm::vec3	gradientW2 = kernel->GetGradientW(positionDirection2);
					const glm::vec3	gradientW3 = kernel->GetGradientW(positionDirection3);
					const glm::vec3	gradientW4 = kernel->GetGradientW(positionDirection4);

					const float volume = 0.25f * boundaryVolume;

					const glm::vec3 velocity1 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot({ 0.0f, 0.0f, 0.0f }, positionDirection1) / (glm::length2(positionDirection1) + 0.01f * info->SupportRadius2) * gradientW1;
					const glm::vec3 velocity2 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot({ 0.0f, 0.0f, 0.0f }, positionDirection2) / (glm::length2(positionDirection2) + 0.01f * info->SupportRadius2) * gradientW2;
					const glm::vec3 velocity3 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot({ 0.0f, 0.0f, 0.0f }, positionDirection3) / (glm::length2(positionDirection3) + 0.01f * info->SupportRadius2) * gradientW3;
					const glm::vec3 velocity4 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot({ 0.0f, 0.0f, 0.0f }, positionDirection4) / (glm::length2(positionDirection4) + 0.01f * info->SupportRadius2) * gradientW4;
					velocity += velocity1 + velocity2 + velocity3 + velocity4;
				}
			}
		}
	}

	b[i] = particleVelocity - info->TimeStepSize / particles[i].Density * velocity;
	g[i] = particleVelocity + particles[i].ViscosityDifference;
}

__global__ void ComputeMatrixVecProdFunctionKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet, 
	vfd::RigidBodyDeviceData** rigidBodies,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::vec3* rhs, 
	glm::vec3* result
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = rhs[i];
	glm::vec3 particleAcceleration = { 0.0f, 0.0f, 0.0f };

	for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);

		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		const glm::vec3& neighborParticleVelocity = rhs[neighborIndex];
		const float neighborParticleDensity = particles[neighborIndex].Density;

		const glm::vec3 positionDirection = particlePosition - neighborParticlePosition;
		const glm::vec3 gradientW = kernel->GetGradientW(positionDirection);

		particleAcceleration += 10.0f * info->DynamicViscosity * (info->ParticleMass / neighborParticleDensity) * glm::dot(particleVelocity - neighborParticleVelocity, positionDirection) / (glm::length2(positionDirection) + 0.01f * info->SupportRadius2) * gradientW;
	}

	if(info->DynamicBoundaryViscosity != 0.0f)
	{
		for (unsigned int j = 0u; j < info->RigidBodyCount; j++)
		{
			vfd::RigidBodyDeviceData* rigidBody = rigidBodies[j];
			const float boundaryVolume = rigidBody->BoundaryVolume[i];

			if (boundaryVolume > 0.0f) {
				const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
				const glm::vec3 positionDirection = particlePosition - neighborPosition;
				glm::vec3 normal = -positionDirection;
				const float normalLength = glm::length(normal);

				if (normalLength > 0.0001f)
				{
					normal /= normalLength;

					glm::vec3 t1;
					glm::vec3 t2;
					vfd::GetOrthogonalVectors(normal, t1, t2);

					const glm::vec3 position1 = neighborPosition - t1 * info->TangentialDistance;
					const glm::vec3 position2 = neighborPosition + t1 * info->TangentialDistance;
					const glm::vec3 position3 = neighborPosition - t2 * info->TangentialDistance;
					const glm::vec3 position4 = neighborPosition + t2 * info->TangentialDistance;

					const glm::vec3 positionDirection1 = particlePosition - position1;
					const glm::vec3	positionDirection2 = particlePosition - position2;
					const glm::vec3	positionDirection3 = particlePosition - position3;
					const glm::vec3	positionDirection4 = particlePosition - position4;

					const glm::vec3	gradientW1 = kernel->GetGradientW(positionDirection1);
					const glm::vec3	gradientW2 = kernel->GetGradientW(positionDirection2);
					const glm::vec3	gradientW3 = kernel->GetGradientW(positionDirection3);
					const glm::vec3	gradientW4 = kernel->GetGradientW(positionDirection4);

					const float volume = 0.25f * boundaryVolume;

					const glm::vec3 acceleration1 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot(particleVelocity, positionDirection1) / (glm::length2(positionDirection1) + 0.01f * info->SupportRadius2) * gradientW1;
					const glm::vec3 acceleration2 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot(particleVelocity, positionDirection2) / (glm::length2(positionDirection2) + 0.01f * info->SupportRadius2) * gradientW2;
					const glm::vec3 acceleration3 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot(particleVelocity, positionDirection3) / (glm::length2(positionDirection3) + 0.01f * info->SupportRadius2) * gradientW3;
					const glm::vec3 acceleration4 = 10.0f * info->DynamicBoundaryViscosity * volume * glm::dot(particleVelocity, positionDirection4) / (glm::length2(positionDirection4) + 0.01f * info->SupportRadius2) * gradientW4;
					particleAcceleration += acceleration1 + acceleration2 + acceleration3 + acceleration4;
				}
			}
		}
	}

	result[i] = rhs[i] - info->TimeStepSize / particles[i].Density * particleAcceleration;
}

__global__ void ApplyViscosityForceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	glm::vec3* x
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& newVi = x[i];
	const glm::vec3& velocity = particles[i].Velocity;

	particles[i].Acceleration += 1.0f / info->TimeStepSize * (newVi - velocity);
	particles[i].ViscosityDifference = newVi - velocity;
}

__device__ bool ClassifyParticleConfigurable(
	const vfd::DFSPHSimulationInfo* info,
	float com,
	unsigned int non, 
	float offset)
{
	const float neighborsOnTheLine = info->ClassifierSlope * com + info->ClassifierConstant + offset;
	return static_cast<float>(non) <= neighborsOnTheLine;
}

__device__ void Normalize(glm::vec3& value)
{
	if (glm::length2(value) > 0.0f) {
		value = glm::normalize(value);
	}
}

__global__ void ComputeSurfaceTensionClassificationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particleSurfaceNormals = particles[i].MonteCarloSurfaceNormal = { 0.0f, 0.0f, 0.0f };
	const glm::vec3& particlePosition = particles[i].Position;
	const unsigned int neighborCount = pointSet->GetNeighborCount(i);
	glm::vec3 centerOfMass = { 0.0f, 0.0f, 0.0f };

	if(neighborCount == 0u)
	{
		particles[i].MonteCarloSurfaceCurvature = 1.0f / info->SupportRadius;
		return;
	}

	for (unsigned int j = 0u; j < neighborCount; j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		centerOfMass += particles[neighborIndex].Position - particlePosition;
	}

	centerOfMass /= info->SupportRadius;
	const float classifierInput = glm::length(centerOfMass) / static_cast<float>(neighborCount);

	if(ClassifyParticleConfigurable(info, classifierInput, neighborCount))
	{
		unsigned int particleCount = 0u;
		const unsigned int s = i * info->SurfaceTensionSampleCount / 3 * 3;

		// Remove samples covered by neighbors 
		for (unsigned int p = 0u; p < info->SurfaceTensionSampleCount; p++)
		{
			const unsigned int i3 = s + 3 * p;
			constexpr unsigned int mod = 49152u; // haltonVec323 size
			const glm::vec3 point = info->SupportRadius * glm::vec3(haltonVec[i3 % mod], haltonVec[(i3 + 1) % mod], haltonVec[(i3 + 2) % mod]);

			for (unsigned int j = 0u; j < neighborCount; j++)
			{
				const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
				const glm::vec3 positionDirection = particles[neighborIndex].Position - particlePosition;

				glm::vec3 vec = point - positionDirection;
				const float radiusRatio = info->NeighborParticleRadius / info->ParticleRadius;

				if (glm::length2(vec) <= radiusRatio * radiusRatio * info->SupportRadius2)
				{
					goto skipNeighborSimplificationIteration;
				}
			}

			particleSurfaceNormals += point;
			particleCount++;
			skipNeighborSimplificationIteration:;
		}

		if(particleCount > 0)
		{
			Normalize(particleSurfaceNormals);

			particles[i].MonteCarloSurfaceCurvature = 1.0f / info->SupportRadius * -2.0f * sqrt(1.0f - info->NeighborParticleRadius * info->NeighborParticleRadius / (info->ParticleRadius * info->ParticleRadius)) *
				cos(acos(1.0f - 2.0f * (static_cast<float>(particleCount) / static_cast<float>(info->SurfaceTensionSampleCount))) + info->MonteCarloFactor);
		}
		else
		{
			particleSurfaceNormals = { 0.0f, 0.0f, 0.0f };
			particles[i].MonteCarloSurfaceCurvature = 0.0f;
		}
	}
}

__global__ void ComputeSurfaceTensionNormalsAndCurvatureKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	if (particles[i].MonteCarloSurfaceNormal != glm::vec3(0.0f, 0.0f, 0.0f))
	{
		const glm::vec3& particlePosition = particles[i].Position;
		glm::vec3 normalCorrection = { 0.0f, 0.0f, 0.0f };
		float correctionForCurvature = 0.0f;
		float correctionFactor = 0.0f;

		for (unsigned int j = 0u; j < pointSet->GetNeighborCount(i); j++)
		{
			const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
			const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
			const glm::vec3& neighborParticleSurfaceNormal = particles[neighborIndex].MonteCarloSurfaceNormal;

			if(neighborParticleSurfaceNormal != glm::vec3(0.0f, 0.0f, 0.0f))
			{
				const glm::vec3 positionDirection = neighborParticlePosition - particlePosition;
				const float neighborDistance = glm::length(positionDirection);
				const float numerator = 1 - neighborDistance / info->SupportRadius;

				normalCorrection += neighborParticleSurfaceNormal * numerator;
				correctionForCurvature += particles[neighborIndex].MonteCarloSurfaceCurvature * numerator;
				correctionFactor += numerator;
			}
		}

		Normalize(normalCorrection);
		particles[i].MonteCarloSurfaceNormalSmooth = (1 - info->SmoothingFactor) * particles[i].MonteCarloSurfaceNormal + info->SmoothingFactor * normalCorrection;
		Normalize(particles[i].MonteCarloSurfaceNormalSmooth);

		particles[i].MonteCarloSurfaceCurvatureSmooth =
			((1.0f - info->SmoothingFactor) * particles[i].MonteCarloSurfaceCurvature + info->SmoothingFactor * correctionForCurvature) /
			(1.0f - info->SmoothingFactor + info->SmoothingFactor * correctionFactor);
	}
}

__global__ void ComputeSurfaceTensionBlendingKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	if (particles[i].MonteCarloSurfaceNormal != glm::vec3(0.0f, 0.0f, 0.0f))
	{
		const glm::vec3& finalNormal = particles[i].MonteCarloSurfaceNormalSmooth;
		float finalCurvature = particles[i].MonteCarloSurfaceCurvatureSmooth;

		if (info->TemporalSmoothing)
		{
			finalCurvature = 0.05f * finalCurvature + 0.95f * particles[i].DeltaFinalCurvature;
		}

		glm::vec3 force = finalNormal * info->SurfaceTension * finalCurvature;
		glm::vec3& acceleration = particles[i].Acceleration;
		acceleration -= info->ParticleMassInverse * force;

		particles[i].DeltaFinalCurvature = finalCurvature;
	}
	else
	{
		float finalCurvature = 0.0f;
		if (info->TemporalSmoothing)
		{
			finalCurvature = 0.95f * particles[i].DeltaFinalCurvature;
		}

		particles[i].DeltaFinalCurvature = finalCurvature;
	}
}