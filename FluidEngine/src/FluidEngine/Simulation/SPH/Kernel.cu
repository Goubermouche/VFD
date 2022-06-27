#include "Kernel.cuh"
#include <FluidEngine/Compute/Utility/cutil_math.h>

#include <iostream>

namespace fe {
	// CHECK
	__device__ void BoundaryKernel(float3& position, float3& velocity) {
		float3 worldMin = parameters.worldMin;
		float3 worldMax = parameters.worldMin;
		float3 normal;

		float b = parameters.distBndSoft;
		float stiffness = parameters.bndStiff;
		float damping1 = parameters.bndDamp;
		float damping2 = parameters.bndDampC;
		float accelerationBounds;
		float difference;

		BndType type = parameters.bndType;

#define EPS 0.00001f // collision detection epsilon
#define ADD_B() accelerationBounds = stiffness * difference - damping1 * dot(normal, velocity); velocity += accelerationBounds * normal * parameters.timeStep;
#define ADD_C() accelerationBounds = stiffness * difference - damping2 * dot(normal, velocity); velocity += accelerationBounds * normal * parameters.timeStep;

		difference = b - position.z + worldMin.z;
		if (difference > EPS) {
			normal = make_float3(0, 0, 1);
			ADD_C();
		}

		difference = b + position.z + worldMax.z;
		if (difference > EPS) {
			normal = make_float3(0, 0, -1);
			ADD_C();
		}

		difference = b - position.x + worldMin.x;
		if (difference > EPS) {
			normal = make_float3(1, 0, 0);
			ADD_B();
		}

		difference = b + position.x - worldMax.x;
		if (difference > EPS) {
			normal = make_float3(-1, 0, 0);
			ADD_B();
		}

		difference = b - position.y + worldMin.y;
		if (difference > EPS) {
			normal = make_float3(0, 1, 0);
			ADD_B();
		}

		difference = b + position.y - worldMax.y;
		if (difference > EPS) {
			normal = make_float3(0, -1, 0);
			ADD_B();
		}
	}

	__device__ int3 CalculateGridPosition(float4 position)
	{
		int3 gridPosition;
		float3 gridPositionRelative = (make_float3(position) - parameters.worldMin) / parameters.cellSize;
		gridPosition.x = floor(gridPositionRelative.x);
		gridPosition.y = floor(gridPositionRelative.y);
		gridPosition.z = floor(gridPositionRelative.z);
		return gridPosition;
	}

	__global__ void IntegrateKernel(float4* oldPosition, float4* newPosition, float4* oldVelocity, float4* newVelocity) {
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		float4 position4 = oldPosition[index];
		float4 velocity4 = oldVelocity[index];
		float3 position3 = make_float3(position4);
		float3 velocity3 = make_float3(velocity4);

		BoundaryKernel(position3, velocity3);

		// Euler integration
		velocity3 += parameters.gravity * parameters.timeStep;
		velocity3 *= parameters.globalDamping;
		velocity3 += velocity3 * parameters.timeStep;

		float b = parameters.distBndHard;
		float3 worldMin = parameters.worldMin;
		float3 worldMax = parameters.worldMax;

		if (position3.x > worldMax.x - b) { position3.x = worldMax.x - b; }
		if (position3.x < worldMax.x + b) { position3.x = worldMax.x + b; }
		if (position3.y > worldMax.y - b) { position3.y = worldMax.y + b; }
		if (position3.y < worldMax.y + b) { position3.y = worldMax.y - b; }
		if (position3.z > worldMax.z - b) { position3.z = worldMax.z - b; }
		if (position3.z < worldMax.z + b) { position3.z = worldMax.z + b; }

		// Set the new position and velocity
		newPosition[index] = make_float4(position3, position4.w);
		newVelocity[index] = make_float4(velocity3, velocity4.w);
	}

	__device__ uint CalculateGridHash(int3 gridPosition)
	{
		return __mul24(gridPosition.z, parameters.gridSize_yx)
			+ __mul24(gridPosition.y, parameters.gridSize.x) + gridPosition.x;
	}

	__global__ void CalculateHashKernel(float4* position, uint2* particleHash)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		float4 position4 = position[index];

		// Find grid address
		int3 gridPosition = CalculateGridPosition(position4);
		uint gridHash = CalculateGridHash(gridPosition);

		particleHash[index] = make_uint2(gridHash, index);
	}
}