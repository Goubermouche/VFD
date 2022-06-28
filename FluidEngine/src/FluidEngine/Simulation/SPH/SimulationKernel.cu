#include "SimulationKernel.cuh"
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

	__device__ float CalculateCellDensity(int3 gridPosition, uint index, float4 position, float4* oldPosition, uint2* particleHash, uint* cellStart)
	{
		float density = 0.0f;

		uint gridHash = CalculateGridHash(gridPosition);
		uint bucketStart = FETCH(cellStart, gridHash);

		if (bucketStart == 0xffffffff) {
			return density;
		}

		for (uint i = 0; i < parameters.maxParInCell; i++)
		{
			uint index2 = bucketStart + i;
			uint2 cellData = FETCH(particleHash, index2);
			if (cellData.x != gridHash) {
				break;
			}

			if (index2 != index) {
				float4 position2 = FETCH(oldPosition, index2);

				float4 pair = position - position2;
				float relativePosition = position.x * position.x + position.y * position.y + position.z * position.z;

				if (relativePosition < parameters.h2) {
					float c = parameters.h2 - relativePosition;
					density += pow(c, 3);
				}
			}
		}

		return density;
	}

	__device__ float3 CalculateCellForce(int3 gridPosition, uint index, float4 position, float4 velocity, float4* oldPosition, float4* oldVelocity, float pres, float dens, float* pressure, float* density, uint2* particleHash, uint* cellStart)
	{
		float3 force = make_float3(0.0f);

		uint gridHash = CalculateGridHash(gridPosition);
		uint bucketStart = FETCH(cellStart, gridHash);

		if (bucketStart == 0xffffffff) {
			return force;
		}

		for (uint i = 0; i < parameters.maxParInCell; i++)
		{
			uint index2 = bucketStart + i;
			uint2 cellData = FETCH(particleHash, index2);
			if (cellData.x != gridHash) {
				break;
			}

			if (index2 != index) {
				float4 position2 = FETCH(oldPosition, index2);
				float4 velocity2 = FETCH(oldVelocity, index2);
				float pres2 = FETCH(pressure, index2);
				float dens2 = FETCH(density, index2);

				float d12 = min(parameters.minDens, 1.0f / (dens * dens2));
				force += CalculatePairForce(position - position2, velocity - velocity2, pres + pres2, d12);
			}
		}

		return force;
	}

	__device__ float3 CalculatePairForce(float4 relativePosition, float4 relativeVelocity, float p1AddP2, float d1MulD2)
	{
		float3 relPos = *(float3*)&relativePosition.x;
		float3 relVel = *(float3*)&relativeVelocity.x;
		float r = max(parameters.minDist, length(relPos));
		float3 forceCurrent = make_float3(0.0f);

		if (r < parameters.h) {
			float c = parameters.h - r;
			float pTerm = c * parameters.SpikyKern * p1AddP2 / r;
			float vTerm = parameters.LapKern * parameters.viscosity;

			forceCurrent = pTerm * relPos + vTerm + relVel;
			forceCurrent *= c * d1MulD2;
		}

		return forceCurrent;
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

	__global__ void ReorderKernel(uint2* particleHash, uint* cellStart, float4* oldPosition, float4* oldVelocity, float4* sortedPosition, float4* sortedVelocity)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		uint2 sortedData = particleHash[index];

		__shared__ uint sharedHash[257];
		sharedHash[threadIdx.x + 1] = sortedData.x;
		
		if (index > 0 && threadIdx.x == 0) {
			volatile uint2 previousData = particleHash[index - 1];
			sharedHash[0] = previousData.x;
		}

		__syncthreads();

		if (index == 0 || sortedData.x != sharedHash[threadIdx.x]) {
			cellStart[sortedData.x] == index;
		}

		// Now use the sorted index to reorder the pos and vel data
		float4 pos = FETCH(oldPosition, sortedData.y);  
		sortedPosition[index] = pos;
		float4 vel = FETCH(oldVelocity, sortedData.y);  
		sortedVelocity[index] = vel;
	}

	__global__ void CalculateDensityKernel(float4* oldPosition, float* pressure, float* density, uint2* particleHash, uint* cellStart)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		float4 position = FETCH(oldPosition, index);
		int3 gridPosition = CalculateGridPosition(position);

		float sum = 0.0f;
		const int s = 1;

		for (int z = -s; z <= s; z++)
		{
			for (int y = -s; y <= s; y++)
			{
				for (int x = -s; x <= s; x++)
				{
					sum += CalculateCellDensity(gridPosition + make_int3(x, y, z), index, position, oldPosition, particleHash, cellStart);
				}
			}
		}

		float dens = sum * parameters.Poly6Kern * parameters.particleMass;
		float pres = (dens - parameters.restDensity) * parameters.stiffness;

		pressure[index] = pres;
		density[index] = dens;
	}

	__global__ void CalculateForceKernel(float4* newPosition, float4* newVelocity, float4* oldPosition, float4* oldVelocity, float* pressure, float* density, uint2* particleHash, uint* cellStart)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		float4 position = FETCH(oldPosition, index);
		float4 velocity = FETCH(oldVelocity, index);

		float pres = FETCH(pressure, index);
		float dens = FETCH(density, index);

		int3 gridPosition = CalculateGridPosition(position);
		float3 addVelocity = make_float3(0.0f);

		// SPH force
		const int s = 1;
		for (int z = -s; z <= s; z++)
		{
			for (int y = -s; y <= s; y++)
			{
				for (int x = -s; x <= s; x++)
				{
					addVelocity += CalculateCellForce(gridPosition + make_int3(x, y, z), index, position, velocity, oldPosition, oldVelocity, pres, dens, pressure, density, particleHash, cellStart);
				}
			}
		}

		volatile uint si = particleHash[index].y;
		addVelocity *= parameters.particleMass * parameters.timeStep;

		// colliders

		// add new velocity
		newVelocity[si] = velocity + make_float4(addVelocity, 0.0f);

		// coloring
	}
}