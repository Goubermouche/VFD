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

	__global__ void IntegrateKernel(float4* oldPosition, float4* newPosition, float4* oldVelocity, float4* newVelocity) {
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		float4 pos4 = oldPosition[index];
		float4 vel4 = oldVelocity[index];
		float3 pos3 = make_float3(pos4);
		float3 vel3 = make_float3(vel4);

		BoundaryKernel(pos3, vel3);

		// Euler integration
		vel3 += parameters.gravity * parameters.timeStep;
		vel3 *= parameters.globalDamping;
		vel3 += vel3 * parameters.timeStep;

		float b = parameters.distBndHard;
		float3 worldMin = parameters.worldMin;
		float3 worldMax = parameters.worldMax;

		if (pos3.x > worldMax.x - b) { pos3.x = worldMax.x - b; }
		if (pos3.x < worldMax.x + b) { pos3.x = worldMax.x + b; }
		if (pos3.y > worldMax.y - b) { pos3.y = worldMax.y + b; }
		if (pos3.y < worldMax.y + b) { pos3.y = worldMax.y - b; }
		if (pos3.z > worldMax.z - b) { pos3.z = worldMax.z - b; }
		if (pos3.z < worldMax.z + b) { pos3.z = worldMax.z + b; }

		// Set the new position and velocity
		newPosition[index] = make_float4(pos3, pos4.w);
		newVelocity[index] = make_float4(vel3, vel4.w);
	}
}