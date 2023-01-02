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

// OK
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

// OK
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

// OK
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
	const float vj = rigidBody->BoundaryVolume[i];
	if (vj > 0.0f)
	{
		const glm::vec3& xj = rigidBody->BoundaryXJ[i];

		delta += vj * glm::dot(vi, kernel->GetGradientW(xi - xj));
	}

	densityAdvection = density / info->Density0 + info->TimeStepSize * delta;
	particles[i].Factor *= info->TimeStepSize2Inverse;
	const float si = 1.0f - densityAdvection;
	const float residuum = glm::min(si, 0.0f);
	particles[i].PressureRho2 = -residuum * particles[i].Factor;
}

// ~OK
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

// ~OK
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

	glm::vec3& ai = particles[i].PressureAcceleration;
	ai = { 0.0f, 0.0f, 0.0f };

	const float pRho2I = particles[i].PressureRho2V;
	const glm::vec3& xi = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const float pRho2J = particles[neighborIndex].PressureRho2V;
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

// ~OK
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

// ~OK
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

	const float pRho2I = particles[i].PressureRho2V;
	const glm::vec3& xi = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const float pRho2J = particles[neighborIndex].PressureRho2V;
		const float pSum = pRho2I + info->Density0 / info->Density0 * pRho2J;

		//if(pSum != 0.0f)
		//{
		//	printf("%.3f ", pRho2I); // should be < 1, is > 3
		//}

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
	const float d = 10.0f;

	const float h = info->SupportRadius;
	const float h2 = h * h;
	const float dt = info->TimeStepSize;
	const float mu = info->Viscosity * info->Density0;
	const float mub = info->BoundaryViscosity * info->Density0;
	const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;

	const float density_i = particles[i].Density;
	const glm::vec3& xi = particles[i].Position;

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const float density_j = particles[neighborIndex].Density;
		const glm::vec3 gradW = kernel->GetGradientW(xi - xj);
		const glm::vec3 xixj = xi - xj;
		result += d * mu * (particles[neighborIndex].Mass / density_j) / (glm::length2(xixj) + 0.01f * h2) * glm::outerProduct(xixj, gradW);
	}

	if(mub != 0.0f)
	{
		// TODO: Add support for multiple rigid bodies
		const float vj = rigidBody->BoundaryVolume[i];
		if (vj > 0.0f) {
			const glm::vec3& xj = rigidBody->BoundaryXJ[i];

			const glm::vec3 xixj = xi - xj;
			glm::vec3 normal = -xixj;
			const float nl = glm::length(normal);

			if(nl > 0.0001f)
			{
				normal /= nl;

				glm::vec3 t1;
				glm::vec3 t2;
				vfd::GetOrthogonalVectors(normal, t1, t2);

				const float dist = info->TangentialDistanceFactor * h;
				const glm::vec3 x1 = xj - t1 * dist;
				const glm::vec3 x2 = xj + t1 * dist;
				const glm::vec3 x3 = xj - t2 * dist;
				const glm::vec3 x4 = xj + t2 * dist;

				const glm::vec3 xix1 = xi - x1;
				const glm::vec3	xix2 = xi - x2;
				const glm::vec3	xix3 = xi - x3;
				const glm::vec3	xix4 = xi - x4;

				const glm::vec3	gradW1 = kernel->GetGradientW(xix1);
				const glm::vec3	gradW2 = kernel->GetGradientW(xix2);
				const glm::vec3	gradW3 = kernel->GetGradientW(xix3);
				const glm::vec3	gradW4 = kernel->GetGradientW(xix4);

				const float vol = 0.25f * vj;

				result += d * mub * vol / (glm::length2(xix1) + 0.01f * h2) * glm::outerProduct(xix1, gradW1);
				result += d * mub * vol / (glm::length2(xix2) + 0.01f * h2) * glm::outerProduct(xix1, gradW2);
				result += d * mub * vol / (glm::length2(xix3) + 0.01f * h2) * glm::outerProduct(xix1, gradW3);
				result += d * mub * vol / (glm::length2(xix4) + 0.01f * h2) * glm::outerProduct(xix1, gradW4);
			}
		}
	}

	inverseDiagonal[i] = glm::inverse(glm::identity<glm::mat3x3>() - (dt / density_i) * result);
}

__global__ void ComputeViscosityGradientRHSKernel(
	vfd::DFSPHParticle* particles,
	vfd::DFSPHSimulationInfo* info,
	vfd::RigidBodyDeviceData* rigidBody,
	vfd::PrecomputedDFSPHCubicKernel* kernel,
	float* b,
	float* g
)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const float d = 10.0f;

	const float h = info->SupportRadius;
	const float h2 = h * h;
	const float dt = info->TimeStepSize;
	const float mu = info->Viscosity * info->Density0;
	const float mub = info->BoundaryViscosity * info->Density0;
	const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;

	const glm::vec3& vi = particles[i].Velocity;
	const glm::vec3& xi = particles[i].Position;
	const float density_i = particles[i].Density;
	const float m_i = particles[i].Mass;
	glm::vec3 bi = { 0.0f, 0.0f, 0.0f };

	if(mub != 0.0f)
	{
		// TODO: Add support for multiple rigid bodies
		const float vj = rigidBody->BoundaryVolume[i];
		if (vj > 0.0f) {
			const glm::vec3& xj = rigidBody->BoundaryXJ[i];

			const glm::vec3 xixj = xi - xj;
			glm::vec3 normal = -xixj;
			const float nl = glm::length(normal);

			if (nl > 0.0001f)
			{
				normal /= nl;

				glm::vec3 t1;
				glm::vec3 t2;
				vfd::GetOrthogonalVectors(normal, t1, t2);

				const float dist = info->TangentialDistanceFactor * h;
				const glm::vec3 x1 = xj - t1 * dist;
				const glm::vec3 x2 = xj + t1 * dist;
				const glm::vec3 x3 = xj - t2 * dist;
				const glm::vec3 x4 = xj + t2 * dist;

				const glm::vec3 xix1 = xi - x1;
				const glm::vec3	xix2 = xi - x2;
				const glm::vec3	xix3 = xi - x3;
				const glm::vec3	xix4 = xi - x4;

				const glm::vec3	gradW1 = kernel->GetGradientW(xix1);
				const glm::vec3	gradW2 = kernel->GetGradientW(xix2);
				const glm::vec3	gradW3 = kernel->GetGradientW(xix3);
				const glm::vec3	gradW4 = kernel->GetGradientW(xix4);

				const float vol = 0.25f * vj;

				const glm::vec3 a1 = d * mub * vol * glm::dot({ 0.0f, 0.0f, 0.0f }, xix1) / (glm::length2(xix1) + 0.01f * h2) * gradW1;
				const glm::vec3 a2 = d * mub * vol * glm::dot({ 0.0f, 0.0f, 0.0f }, xix2) / (glm::length2(xix2) + 0.01f * h2) * gradW2;
				const glm::vec3 a3 = d * mub * vol * glm::dot({ 0.0f, 0.0f, 0.0f }, xix3) / (glm::length2(xix3) + 0.01f * h2) * gradW3;
				const glm::vec3 a4 = d * mub * vol * glm::dot({ 0.0f, 0.0f, 0.0f }, xix4) / (glm::length2(xix4) + 0.01f * h2) * gradW4;
				bi += a1 + a2 + a3 + a4;
			}
		}
	}

	b[3 * i + 0] = vi[0] - dt / density_i * bi[0];
	b[3 * i + 1] = vi[1] - dt / density_i * bi[1];
	b[3 * i + 2] = vi[2] - dt / density_i * bi[2];

	g[3 * i + 0] = vi[0] + particles[i].ViscosityDifference[0];
	g[3 * i + 1] = vi[1] + particles[i].ViscosityDifference[1];
	g[3 * i + 2] = vi[2] + particles[i].ViscosityDifference[2];
}

__global__ void ComputeMatrixVecProdFunctionKernel(vfd::DFSPHParticle* particles, vfd::DFSPHSimulationInfo* info, const vfd::NeighborSet* pointSet, vfd::RigidBodyDeviceData* rigidBody, vfd::PrecomputedDFSPHCubicKernel* kernel, float* rhs, float* result)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= info->ParticleCount)
	{
		return;
	}

	const float h = info->SupportRadius;
	const float h2 = h * h;
	const float dt = info->TimeStepSize;
	const float mu = info->Viscosity * info->Density0;
	const float mub = info->BoundaryViscosity * info->Density0;
	const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;
	const float d = 10.0f;

	const glm::vec3& xi = particles[i].Position;
	glm::vec3 ai = { 0.0f, 0.0f, 0.0f };
	const float density_i = particles[i].Density;
	const glm::vec3& vi = glm::vec3(rhs[i * 3 + 0], rhs[i * 3 + 1], rhs[i * 3 + 2]);

	for (unsigned int j = 0; j < pointSet->GetNeighborCount(i); j++)
	{
		const unsigned int neighborIndex = pointSet->GetNeighbor(i, j);
		const glm::vec3& xj = particles[neighborIndex].Position;

		const float density_j = particles[neighborIndex].Density;
		const glm::vec3 gradW = kernel->GetGradientW(xi - xj);
		const glm::vec3& vj = glm::vec3(rhs[neighborIndex * 3 + 0], rhs[neighborIndex * 3 + 1], rhs[neighborIndex * 3 + 2]);
		const glm::vec3 xixj = xi - xj;
		ai += d * mu * (particles[neighborIndex].Mass / density_j) * glm::dot(vi - vj, xixj) / (glm::length2(xixj) + 0.01f * h2) * gradW;
	}

	if(mub != 0.0f)
	{
		// TODO: Add support for multiple rigid bodies
		const float vj = rigidBody->BoundaryVolume[i];
		if (vj > 0.0f) {
			const glm::vec3& xj = rigidBody->BoundaryXJ[i];

			const glm::vec3 xixj = xi - xj;
			glm::vec3 normal = -xixj;
			const float nl = glm::length(normal);

			if (nl > 0.0001f)
			{
				normal /= nl;

				glm::vec3 t1;
				glm::vec3 t2;
				vfd::GetOrthogonalVectors(normal, t1, t2);

				const float dist = info->TangentialDistanceFactor * h;
				const glm::vec3 x1 = xj - t1 * dist;
				const glm::vec3 x2 = xj + t1 * dist;
				const glm::vec3 x3 = xj - t2 * dist;
				const glm::vec3 x4 = xj + t2 * dist;

				const glm::vec3 xix1 = xi - x1;
				const glm::vec3	xix2 = xi - x2;
				const glm::vec3	xix3 = xi - x3;
				const glm::vec3	xix4 = xi - x4;

				const glm::vec3	gradW1 = kernel->GetGradientW(xix1);
				const glm::vec3	gradW2 = kernel->GetGradientW(xix2);
				const glm::vec3	gradW3 = kernel->GetGradientW(xix3);
				const glm::vec3	gradW4 = kernel->GetGradientW(xix4);

				const float vol = 0.25f * vj;

				const glm::vec3 a1 = d * mub * vol * glm::dot(vi, xix1) / (glm::length2(xix1) + 0.01f * h2) * gradW1;
				const glm::vec3 a2 = d * mub * vol * glm::dot(vi, xix2) / (glm::length2(xix2) + 0.01f * h2) * gradW2;
				const glm::vec3 a3 = d * mub * vol * glm::dot(vi, xix3) / (glm::length2(xix3) + 0.01f * h2) * gradW3;
				const glm::vec3 a4 = d * mub * vol * glm::dot(vi, xix4) / (glm::length2(xix4) + 0.01f * h2) * gradW4;
				ai += a1 + a2 + a3 + a4;
			}
		}
	}

	result[3 * i] = rhs[3 * i] - dt / density_i * ai[0];
	result[3 * i + 1] = rhs[3 * i + 1] - dt / density_i * ai[1];
	result[3 * i + 2] = rhs[3 * i + 2] - dt / density_i * ai[2];
}