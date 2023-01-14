#ifndef DFSPH_FUNCTION_OBJECTS_H
#define DFSPH_FUNCTION_OBJECTS_H

#include "Simulation/DFSPH/Structures/DFSPHParticle.h"

#include <thrust/device_vector.h>

struct MaxVelocityMagnitudeUnaryOperator
{
	float TimeStepSize;

	// Calculates the velocity magnitude of a given particle using the provided time step size
	__host__ __device__	float operator()(const vfd::DFSPHParticle& x) const {
		return glm::length2(x.Velocity + x.Acceleration * TimeStepSize);
	}
};

struct DensityErrorUnaryOperator
{
	float Density0;

	__host__ __device__	float operator()(const vfd::DFSPHParticle& x) const {
		return Density0 * x.PressureResiduum;
	}
};

struct SquaredNormUnaryOperator
{
	__host__ __device__	float operator()(const glm::vec3& x) const {
		return glm::compAdd(x * x);
	}
};

struct DotUnaryOperator
{
	__host__ __device__	float operator()(thrust::tuple<glm::vec3, glm::vec3> tuple) const {
		return  glm::compAdd(thrust::get<0>(tuple) * thrust::get<1>(tuple));
	}
};

struct Vec3FloatMultiplyBinaryOperator
{
	__host__ __device__	glm::vec3 operator()(const glm::vec3& x, const float& y) const
	{
		return x * y;
	}
};

struct Vec3Mat3MultiplyBinaryOperator
{
	__host__ __device__	glm::vec3 operator()(const glm::mat3x3& x, const glm::vec3& y) const
	{
		return x * y;
	}
};

#endif // !DFSPH_FUNCTION_OBJECTS_H