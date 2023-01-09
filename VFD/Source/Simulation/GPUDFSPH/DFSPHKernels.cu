#include "pch.h"
#include "DFSPHKernels.cuh"

__global__ void ClearAccelerationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= info->ParticleCount)
	{
		return;
	}

	if(particles[i].Mass != 0.0f)
	{
		particles[i].Acceleration = info->Gravity;
	}
}

__global__ void ComputeVelocityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

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
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Position += info->TimeStepSize * particles[i].Velocity;
}

__global__ void ComputeVolumeAndBoundaryKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData* rigidBody
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePosition = particles[i].Position;
	glm::vec3& boundaryPosition = rigidBody->BoundaryXJ[i] = { 0.0f, 0.0f, 0.0f };
	float& boundaryVolume = rigidBody->BoundaryVolume[i] = 0.0f;

	glm::dvec3 normal;

	double distance = DBL_MAX;
	glm::dvec3 interpolationVector;
	unsigned int cell[32];
	double shapeFunction[32];
	glm::dvec3 shapeFunctionDerivative[32];

	if (rigidBody->Map->DetermineShapeFunction(0, particlePosition, cell, interpolationVector, shapeFunction, shapeFunctionDerivative))
	{
		distance = rigidBody->Map->Interpolate(0, cell, interpolationVector, shapeFunction, normal, shapeFunctionDerivative);
	}

	if (distance > 0.0 && static_cast<float>(distance) < info->SupportRadius)
	{
		const double volume = rigidBody->Map->Interpolate(1, cell, interpolationVector, shapeFunction);
		if (volume > 0.0 && volume != DBL_MAX)
		{
			boundaryVolume = static_cast<float>(volume);
			const double normalLength = glm::length(normal);

			if (normalLength > 1.0e-9)
			{
				normal /= normalLength;
				const float particleDistance = glm::max((static_cast<float>(distance) + 0.5f * info->ParticleRadius), info->ParticleDiameter);
				boundaryPosition = particlePosition - particleDistance * static_cast<glm::vec3>(normal);
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
	else if (distance <= 0.0)
	{
		if (distance != DBL_MAX)
		{
			const double normalLength = glm::length(normal);

			if (normalLength > 1.0e-5)
			{
				normal /= normalLength;
				float delta = info->ParticleDiameter - static_cast<float>(distance);
				delta = glm::min(delta, 0.1f * info->ParticleRadius);

				particlePosition += delta * static_cast<glm::vec3>(normal);
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

__global__ void ComputeDensityKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet, 
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	float& particleDensity = particles[i].Density;
	particleDensity = info->Volume * kernel->GetWZero();

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++) {
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		particleDensity += info->Volume * kernel->GetW(particlePosition - particles[neighborIndex].Position);
	}

	// TODO: Add support for multiple rigid bodies
	const float boundaryVolume = rigidBody->BoundaryVolume[i];

	if (boundaryVolume > 0.0f) {
		particleDensity += boundaryVolume * kernel->GetW(particlePosition - rigidBody->BoundaryXJ[i]);
	}

	particleDensity *= info->Density0;
}

__global__ void ComputeDFSPHFactorKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody, 
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	glm::vec3 gradientPI = { 0.0f, 0.0f, 0.0f };
	float sumGradientPK = 0.0f;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
		sumGradientPK += glm::dot(gradientPJ, gradientPJ);
		gradientPI -= gradientPJ;
	}

	// TODO: Add support for multiple rigid bodies
	const float boundaryVolume = rigidBody->BoundaryVolume[i];

	if(boundaryVolume > 0.0f)
	{
		const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
		const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);
		gradientPI -= gradientPJ;
	}

	sumGradientPK += glm::dot(gradientPI, gradientPI);
	particles[i].Factor = sumGradientPK > EPS ? particles[i].Factor = 1.0f / sumGradientPK : 0.0f;
}

__global__ void ComputeDensityAdvectionKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody, 
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = particles[i].Velocity;
	float& particleDensityAdvection = particles[i].DensityAdvection;
	float delta = 0.0f;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		delta += glm::dot(particleVelocity - particles[neighborIndex].Velocity, kernel->GetGradientW(particlePosition - particles[neighborIndex].Position));
	}

	delta *= info->Volume;

	// TODO: Add support for multiple rigid bodies
	const float boundaryVolume = rigidBody->BoundaryVolume[i];

	if (boundaryVolume > 0.0f)
	{
		const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
		delta += boundaryVolume * glm::dot(particleVelocity, kernel->GetGradientW(particlePosition - neighborPosition));
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
	const vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleAcceleration = particles[i].PressureAcceleration;
	float densityPressureForce = 0.0f;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		densityPressureForce += glm::dot(particleAcceleration - particles[neighborIndex].PressureAcceleration, kernel->GetGradientW(particlePosition - neighborParticlePosition));
	}

	densityPressureForce *= info->Volume;

	// TODO: Add support for multiple rigid bodies
	const float boundaryVolume = rigidBody->BoundaryVolume[i];

	if (boundaryVolume > 0.0f)
	{
		const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
		densityPressureForce += boundaryVolume * glm::dot(particleAcceleration, kernel->GetGradientW(particlePosition - neighborPosition));
	}

	return densityPressureForce;
}

__global__ void PressureSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet, 
	const vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const float& particleDensityAdvection = particles[i].DensityAdvection;

	float densityPressureForce = ComputeDensityPressureForce(i, particles, info, pointSet, rigidBody, kernel);
	densityPressureForce *= info->TimeStepSize2;

	float& residuum = particles[i].PressureResiduum;
	residuum = glm::min(1.0f - particleDensityAdvection - densityPressureForce, 0.0f);
	particles[i].PressureRho2 -= residuum * particles[i].Factor;
}

__global__ void ComputePressureAccelerationKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet, 
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const float particlePressureRho = particles[i].PressureRho2;
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const float pressureSum = particlePressureRho + info->Density0 / info->Density0 * particles[neighborIndex].PressureRho2;
		
		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		// TODO: Add support for multiple rigid bodies
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f) {
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

			particlePressureAcceleration += particlePressureRho * gradientPJ;
		}
	}
}

__global__ void ComputePressureAccelerationAndDivergenceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const float particlePressureRho = particles[i].PressureRho2V;
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const float pressureSum = particlePressureRho + info->Density0 / info->Density0 * particles[neighborIndex].PressureRho2V;

		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		// TODO: Add support for multiple rigid bodies
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f) {
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

			particlePressureAcceleration += particlePressureRho * gradientPJ;
		}
	}
}

__global__ void ComputePressureAccelerationAndVelocityKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info, 
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const float particlePressureRho = particles[i].PressureRho2;
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		const float pressureSum = particlePressureRho + info->Density0 / info->Density0 * particles[neighborIndex].PressureRho2;

		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - neighborParticlePosition);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		// TODO: Add support for multiple rigid bodies
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f) {
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

			particlePressureAcceleration += particlePressureRho * gradientPJ;
		}
	}

	particles[i].Velocity += info->TimeStepSize * particlePressureAcceleration;
}

__global__ void ComputeDensityChangeKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	float& particleDensityAdvection = particles[i].DensityAdvection;
	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = particles[i].Velocity;

	particleDensityAdvection = 0.0f;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		particleDensityAdvection += glm::dot(particleVelocity - particles[neighborIndex].Velocity, kernel->GetGradientW(particlePosition - neighborParticlePosition));
	}

	particleDensityAdvection *= info->Volume;

	// TODO: Add support for multiple rigid bodies
	const float boundaryVolume = rigidBody->BoundaryVolume[i];

	if (boundaryVolume > 0.0f) {
		const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
		particleDensityAdvection += boundaryVolume * glm::dot(particleVelocity, kernel->GetGradientW(particlePosition - neighborPosition));
	}

	particleDensityAdvection = pointSet->GetNeighborCount(i) < 20 ? 0.0f : glm::max(particleDensityAdvection, 0.0f);

	float& factor = particles[i].Factor;
	factor *= info->TimeStepSizeInverse;
	particles[i].PressureRho2V = particleDensityAdvection * factor;
}

__global__ void DivergenceSolveIterationKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	const vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	float densityPressureForce = ComputeDensityPressureForce(i, particles, info, pointSet, rigidBody, kernel);
	densityPressureForce *= info->TimeStepSize;

	float& residuum = particles[i].PressureResiduum;
	residuum = pointSet->GetNeighborCount(i) < 20 ? 0.0f : glm::min(-particles[i].DensityAdvection - densityPressureForce, 0.0f);

	particles[i].PressureRho2V -= residuum * particles[i].Factor;
}

__global__ void ComputePressureAccelerationAndFactorKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& particlePressureAcceleration = particles[i].PressureAcceleration = { 0.0f, 0.0f, 0.0f };
	const glm::vec3& particlePosition = particles[i].Position;
	const float particlePressureRho = particles[i].PressureRho2V;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const float pressureSum = particlePressureRho + info->Density0 / info->Density0 * particles[neighborIndex].PressureRho2V;

		if (fabs(pressureSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(particlePosition - particles[neighborIndex].Position);
			particlePressureAcceleration += pressureSum * gradientPJ;
		}
	}

	if (fabs(particlePressureRho) > EPS) {
		// TODO: Add support for multiple rigid bodies
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f) {
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			const glm::vec3 gradientPJ = -boundaryVolume * kernel->GetGradientW(particlePosition - neighborPosition);

			particlePressureAcceleration += particlePressureRho * gradientPJ;
		}
	}

	particles[i].Velocity += info->TimeStepSize * particlePressureAcceleration;
	particles[i].Factor *= info->TimeStepSize;
}

__global__ void ComputeViscosityPreconditionerKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::mat3x3* inverseDiagonal
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::mat3x3 result = glm::mat3x3();
	const glm::vec3& particlePosition = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);

		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		const float neighborParticleDensity = particles[neighborIndex].Density;

		const glm::vec3 positionDirection = particlePosition - neighborParticlePosition;
		const glm::vec3 gradientW = kernel->GetGradientW(positionDirection);

		result += 10.0f * info->DynamicViscosity * (particles[neighborIndex].Mass / neighborParticleDensity) / (glm::length2(positionDirection) + 0.01f * info->SupportRadius2) * glm::outerProduct(positionDirection, gradientW);
	}

	if(info->DynamicBoundaryViscosity != 0.0f)
	{
		// TODO: Add support for multiple rigid bodies
		const float boundaryVolume = rigidBody->BoundaryVolume[i];

		if (boundaryVolume > 0.0f) {
			const glm::vec3& neighborPosition = rigidBody->BoundaryXJ[i];
			const glm::vec3 neighborDirection = particlePosition - neighborPosition;
			glm::vec3 normal = -neighborDirection;
			const float normalLength = glm::length(normal);

			if(normalLength > 0.0001f)
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

	inverseDiagonal[i] = glm::inverse(glm::identity<glm::mat3x3>() - (info->TimeStepSize / particles[i].Density) * result);
}

__global__ void ComputeViscosityGradientKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::vec3* b,
	glm::vec3* g
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = particles[i].Velocity;
	glm::vec3 velocity = { 0.0f, 0.0f, 0.0f };

	if(info->DynamicBoundaryViscosity != 0.0f)
	{
		// TODO: Add support for multiple rigid bodies
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

	b[i] = particleVelocity - info->TimeStepSize / particles[i].Density * velocity;
	g[i] = particleVelocity + particles[i].ViscosityDifference;
}

__global__ void ComputeMatrixVecProdFunctionKernel(
	vfd::DFSPHParticle* particles, 
	vfd::DFSPHSimulationInfo* info,
	const vfd::NeighborSet* pointSet, 
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	glm::vec3* rhs, 
	glm::vec3* result
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& particlePosition = particles[i].Position;
	const glm::vec3& particleVelocity = rhs[i];
	glm::vec3 particleAcceleration = { 0.0f, 0.0f, 0.0f };

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);

		const glm::vec3& neighborParticlePosition = particles[neighborIndex].Position;
		const glm::vec3& neighborParticleVelocity = rhs[neighborIndex];
		const float neighborParticleDensity = particles[neighborIndex].Density;

		const glm::vec3 positionDirection = particlePosition - neighborParticlePosition;
		const glm::vec3 gradientW = kernel->GetGradientW(positionDirection);

		particleAcceleration += 10.0f * info->DynamicViscosity * (particles[neighborIndex].Mass / neighborParticleDensity) * glm::dot(particleVelocity - neighborParticleVelocity, positionDirection) / (glm::length2(positionDirection) + 0.01f * info->SupportRadius2) * gradientW;
	}

	if(info->DynamicBoundaryViscosity != 0.0f)
	{
		// TODO: Add support for multiple rigid bodies
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

	result[i] = rhs[i] - info->TimeStepSize / particles[i].Density * particleAcceleration;
}

__global__ void ApplyViscosityForceKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	glm::vec3* x
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& newVi = x[i];
	const glm::vec3& velocity = particles[i].Velocity;

	particles[i].Acceleration += 1.0f / info->TimeStepSize * (newVi - velocity);
	particles[i].ViscosityDifference = newVi - velocity;
}