#include "pch.h"
#include "DFSPHKernels.cuh"

__global__ void ClearAccelerationKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
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

__global__ void ComputeVelocityKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	particles[i].Velocity += info->TimeStepSize * particles[i].Acceleration;
}

__global__ void ComputePositionKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info)
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

	const glm::vec3& xi = particles[i].Position;
	glm::vec3& boundaryXj = rigidBody->BoundaryXJ[i];
	float& boundaryVolume = rigidBody->BoundaryVolume[i];

	boundaryXj = { 0.0f, 0.0f, 0.0f };
	boundaryVolume = 0.0f;

	glm::vec3 t(0, -0.25, 0);
	glm::dvec3 normal;
	const glm::mat3& rotationMatrix = rigidBody->Rotation;
	const glm::dvec3 localXi = glm::transpose(rotationMatrix) * xi; // TODO: transformation matrix? 

	double dist = DBL_MAX;
	glm::dvec3 c0;
	unsigned int cell[32];
	double N[32];
	glm::dvec3 dN[32];

	if (rigidBody->Map->DetermineShapeFunction(0, localXi, cell, c0, N, dN))
	{
		dist = rigidBody->Map->Interpolate(0, cell, c0, N, normal, dN);
	}

	if (dist > 0.0 && static_cast<float>(dist) < info->SupportRadius)
	{
		const double volume = rigidBody->Map->Interpolate(1, cell, c0, N);
		if (volume > 0.0 && volume != DBL_MAX)
		{
			boundaryVolume = static_cast<float>(volume);
			normal = static_cast<glm::dmat3>(rotationMatrix) * normal;
			const double nl = glm::length(normal);

			if (nl > 1.0e-9)
			{
				normal /= nl;
				const float d = glm::max((static_cast<float>(dist) + 0.5f * info->ParticleRadius), info->ParticleDiameter);
				boundaryXj = xi - d * static_cast<glm::vec3>(normal);
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
	else if (dist <= 0.0)
	{
		if (dist != DBL_MAX)
		{
			normal = static_cast<glm::dmat3>(rotationMatrix) * normal;
			const double nl = glm::length(normal);

			if (nl > 1.0e-5)
			{
				normal /= nl;
				float delta = info->ParticleDiameter - static_cast<float>(dist);
				delta = glm::min(delta, 0.1f * info->ParticleRadius);

				particles[i].Position = xi + delta * static_cast<glm::vec3>(normal);
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

__global__ void ComputeDensityKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	float& density = particles[i].Density;
	density = info->Volume * kernel->GetWZero();
	const glm::vec3& xi = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++) {
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		density += info->Volume * kernel->GetW(xi - xj);
	}

	// TODO: Add support for multiple rigid bodies
	const float vj = rigidBody->BoundaryVolume[i];
	if (vj > 0.0f) {
		const glm::vec3& xj = rigidBody->BoundaryXJ[i];
		density += vj * kernel->GetW(xi - xj);
	}

	density *= info->Density0;
}

__global__ void ComputeDFSPHFactorKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info,const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const glm::vec3& xi = particles[i].Position;
	float sumGradientPK = 0.0f;
	glm::vec3 gradientPI = { 0.0f, 0.0f, 0.0f };

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(xi - xj);
		sumGradientPK += glm::dot(gradientPJ, gradientPJ);
		gradientPI -= gradientPJ;
	}

	// TODO: Add support for multiple rigid bodies
	const float vj = rigidBody->BoundaryVolume[i];
	if(vj > 0.0f)
	{
		const glm::vec3& xj = rigidBody->BoundaryXJ[i];

		const glm::vec3 gradientPJ = -vj * kernel->GetGradientW(xi - xj);
		gradientPI -= gradientPJ;
	}

	sumGradientPK += glm::dot(gradientPI, gradientPI);

	if(sumGradientPK > EPS)
	{
		particles[i].Factor = 1.0f / sumGradientPK;
	}
	else
	{
		particles[i].Factor = 0.0f;
	}
}

__global__ void ComputeDensityAdvectionKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const float& density = particles[i].Density;
	float& densityAdvection = particles[i].DensityAdvection;
	const glm::vec3& xi = particles[i].Position;
	const glm::vec3& vi = particles[i].Velocity;
	float delta = 0.0f;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const glm::vec3& vj = particles[neighborIndex].Velocity;
		delta += glm::dot(vi - vj, kernel->GetGradientW(xi - xj));
	}

	delta *= info->Volume;

	// TODO: Add support for multiple rigid bodies
	const float Vj = rigidBody->BoundaryVolume[i];
	if (Vj > 0.0f)
	{
		const glm::vec3& xj = rigidBody->BoundaryXJ[i];

		glm::vec3 vj = { 0.0f, 0.0f, 0.0f };
		delta += Vj * glm::dot(vi - vj, kernel->GetGradientW(xi - xj));
	}

	densityAdvection = density / info->Density0 + info->TimeStepSize * delta;

	particles[i].Factor *= info->TimeStepSize2Inverse;

	const float si = 1.0f - densityAdvection;
	const float residuum = glm::min(si, 0.0f);
	particles[i].PressureRho2 = -residuum * particles[i].Factor;
}

__device__ float ComputeAIJPJ(const unsigned int i, vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel) {
	const glm::vec3& xi = particles[i].Position;
	const glm::vec3& ai = particles[i].PressureAcceleration;
	float aijPJ = 0.0f;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const glm::vec3& aj = particles[neighborIndex].PressureAcceleration;
		aijPJ += glm::dot(ai - aj, kernel->GetGradientW(xi - xj));
	}

	aijPJ *= info->Volume;

	// TODO: Add support for multiple rigid bodies
	const float vj = rigidBody->BoundaryVolume[i];
	if (vj > 0.0f)
	{
		const glm::vec3& xj = rigidBody->BoundaryXJ[i];

		aijPJ += vj * glm::dot(ai, kernel->GetGradientW(xi - xj));
	}

	return aijPJ;
}

__global__ void PressureSolveIterationKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	float aijPJ = ComputeAIJPJ(i, particles, info, pointSet, rigidBody, kernel);
	aijPJ *= info->TimeStepSize2;

	const float& densityAdv = particles[i].DensityAdvection;
	const float si = 1.0f - densityAdv;

	float& pRho2I = particles[i].PressureRho2;
	float residuum = glm::min(si - aijPJ, 0.0f);
	pRho2I -= residuum * particles[i].Factor;
	particles[i].PressureResiduum = residuum;
}

__global__ void ComputePressureAccelerationKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& ai = particles[i].PressureAcceleration;
	ai = { 0.0f, 0.0f, 0.0f };

	const float pRho2I = particles[i].PressureRho2;
	const glm::vec3& xi = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const float pRho2J = particles[neighborIndex].PressureRho2;
		const float pSum = pRho2I + info->Density0 / info->Density0 * pRho2J;
		
		if (fabs(pSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(xi - xj);
			ai += pSum * gradientPJ;
		}
	}

	if (fabs(pRho2I) > EPS) {
		// TODO: Add support for multiple rigid bodies
		const float vj = rigidBody->BoundaryVolume[i];
		if (vj > 0.0f) {
			const glm::vec3& xj = rigidBody->BoundaryXJ[i];

			const glm::vec3 gradientPJ = -vj * kernel->GetGradientW(xi - xj);
			const glm::vec3 a = 1.0f * pRho2I * gradientPJ;
			ai += a;
		}
	}
}

__global__ void ComputePressureAccelerationAndVelocityKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	glm::vec3& ai = particles[i].PressureAcceleration;
	ai = { 0.0f, 0.0f, 0.0f };

	const float pRho2I = particles[i].PressureRho2;
	const glm::vec3& xi = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const float pRho2J = particles[neighborIndex].PressureRho2;
		const float pSum = pRho2I + info->Density0 / info->Density0 * pRho2J;

		if (fabs(pSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(xi - xj);
			ai += pSum * gradientPJ;
		}
	}

	if (fabs(pRho2I) > EPS) {
		// TODO: Add support for multiple rigid bodies
		const float vj = rigidBody->BoundaryVolume[i];
		if (vj > 0.0f) {
			const glm::vec3& xj = rigidBody->BoundaryXJ[i];

			const glm::vec3 gradientPJ = -vj * kernel->GetGradientW(xi - xj);
			const glm::vec3 a = 1.0f * pRho2I * gradientPJ;
			ai += a;
		}
	}

	particles[i].Velocity += info->TimeStepSize * ai;
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

	float& densityAdvection = particles[i].DensityAdvection;
	const glm::vec3& xi = particles[i].Position;
	const glm::vec3& vi = particles[i].Velocity;

	densityAdvection = 0.0f;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const glm::vec3& vj = particles[neighborIndex].Velocity;
		densityAdvection += glm::dot(vi - vj, kernel->GetGradientW(xi - xj));
	}

	densityAdvection *= info->Volume;

	// TODO: Add support for multiple rigid bodies
	const float vj = rigidBody->BoundaryVolume[i];
	if (vj > 0.0f) {
		const glm::vec3& xj = rigidBody->BoundaryXJ[i];

		densityAdvection += vj * glm::dot(vi, kernel->GetGradientW(xi - xj));
	}

	densityAdvection = glm::max(densityAdvection, 0.0f);
	unsigned int numNeighbors = pointSet->GetNeighborCount(i);

	if(numNeighbors < 20)
	{
		densityAdvection = 0.0f;
	}

	float& factor = particles[i].Factor;
	factor *= info->TimeStepSizeInverse;
	particles[i].PressureRho2V = densityAdvection * factor;
}

__global__ void DivergenceSolveIterationKernel(
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

	float aijPJ = ComputeAIJPJ(i, particles, info, pointSet, rigidBody, kernel);
	aijPJ *= info->TimeStepSize;

	const float densityAdvection = particles[i].DensityAdvection;
	const float si = -densityAdvection;

	float& pvRho2I = particles[i].PressureRho2V;
	float residuum = glm::min(si - aijPJ, 0.0f);

	unsigned int numNeighbors = pointSet->GetNeighborCount(i);

	if(numNeighbors < 20)
	{
		residuum = 0.0f;
	}

	pvRho2I -= residuum * particles[i].Factor;
	particles[i].PressureResiduum = residuum;
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

	glm::vec3& ai = particles[i].PressureAcceleration;
	ai = { 0.0f, 0.0f, 0.0f };

	const float pRho2I = particles[i].PressureRho2;
	const glm::vec3& xi = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const float pRho2J = particles[neighborIndex].PressureRho2;
		const float pSum = pRho2I + info->Density0 / info->Density0 * pRho2J;

		if (fabs(pSum) > EPS) {
			const glm::vec3 gradientPJ = -info->Volume * kernel->GetGradientW(xi - xj);
			ai += pSum * gradientPJ;
		}
	}

	if (fabs(pRho2I) > EPS) {
		// TODO: Add support for multiple rigid bodies
		const float vj = rigidBody->BoundaryVolume[i];
		if (vj > 0.0f) {
			const glm::vec3& xj = rigidBody->BoundaryXJ[i];

			const glm::vec3 gradientPJ = -vj * kernel->GetGradientW(xi - xj);
			const glm::vec3 a = 1.0f * pRho2I * gradientPJ;
			ai += a;
		}
	}

	particles[i].Velocity += info->TimeStepSize * ai;
	particles[i].Factor *= info->TimeStepSize;
}