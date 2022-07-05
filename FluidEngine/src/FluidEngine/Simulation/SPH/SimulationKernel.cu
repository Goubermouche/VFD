#ifndef SIMULATION_KERNEL_CU_
#define SIMULATION_KERNEL_CU_

#include "FluidEngine/Compute/Utility/CUDA/cutil_math.h"
#include "SimulationParameters.cuh"
#include <iostream>

namespace fe {
	texture<float4, 1, cudaReadModeElementType> oldPositionTexture;
	texture<float4, 1, cudaReadModeElementType> oldVelocityTexture;
	texture<uint2, 1, cudaReadModeElementType> particleHashTexture;
	texture<unsigned int, 1, cudaReadModeElementType> cellStartTexture;
	texture<float, 1, cudaReadModeElementType> pressureTexture;
	texture<float, 1, cudaReadModeElementType> densityTexture;

	__constant__ SimulationParameters c_parameters;

	static __device__ void CalculateBoundary(float3& position, float3& velocity)
	{
		float3 worldMin = c_parameters.worldMin;
		float3 worldMax = c_parameters.worldMax;
		float3 normal;

		float bounds = c_parameters.boundsSoftDistance;
		float stiffness = c_parameters.boundsStiffness;
		float damping0 = c_parameters.boundsDamping;
		float damping1 = c_parameters.boundsDampingCritical;
		float acceleration;
		float difference;

#define  EPS	0.00001f // epsilon 
#define  ADD_BOUNDS0()  acceleration = stiffness * difference - damping0 * dot(normal, velocity);  velocity += acceleration * normal * c_parameters.timeStep;
#define  ADD_BOUNDS1()  acceleration = stiffness * difference - damping1 * dot(normal, velocity);  velocity += acceleration * normal * c_parameters.timeStep;

		// Box bounds
		difference = bounds - position.z + worldMin.z;
		if (difference > EPS) { normal = make_float3(0, 0, 1); ADD_BOUNDS1(); }
		difference = bounds + position.z - worldMax.z;
		if (difference > EPS) { normal = make_float3(0, 0, -1); ADD_BOUNDS1(); }
		difference = bounds - position.x + worldMin.x;
		if (difference > EPS) { normal = make_float3(1, 0, 0); ADD_BOUNDS0(); }
		difference = bounds + position.x - worldMax.x;
		if (difference > EPS) { normal = make_float3(-1, 0, 0); ADD_BOUNDS0(); }
		difference = bounds - position.y + worldMin.y;
		if (difference > EPS) { normal = make_float3(0, 1, 0); ADD_BOUNDS0(); }
		difference = bounds + position.y - worldMax.y;
		if (difference > EPS) { normal = make_float3(0, -1, 0); ADD_BOUNDS0(); }
	}

	static __global__ void IntegrateKernel(float4* newPosition, float4* oldPosition, float4* newVelocity, float4* oldVelocity) {
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Load position and velocity into registers
		float4 positionFloat4 = oldPosition[index];
		float4 velocityFloat4 = oldVelocity[index];
		float3 position = make_float3(positionFloat4);
		float3 velocity = make_float3(velocityFloat4);

		// Calculate boundary conditions
		CalculateBoundary(position, velocity);

		// Add gravity force to velocity
		velocity += c_parameters.gravity * c_parameters.timeStep;
		velocity *= c_parameters.globalDamping;

		// Update the position of the particle
		position += velocity * c_parameters.timeStep;

		// Clamp the position to the world boundaries
		float b = c_parameters.boundsHardDistance;
		float3 wmin = c_parameters.worldMin, wmax = c_parameters.worldMax;
		if (position.x > wmax.x - b) { position.x = wmax.x - b; }
		if (position.x < wmin.x + b) { position.x = wmin.x + b; }
		if (position.y > wmax.y - b) { position.y = wmax.y - b; }
		if (position.y < wmin.y + b) { position.y = wmin.y + b; }
		if (position.z > wmax.z - b) { position.z = wmax.z - b; }
		if (position.z < wmin.z + b) { position.z = wmin.z + b; }

		// Stores the new position and velocity of the particle
		newPosition[index] = make_float4(position, positionFloat4.w);
		newVelocity[index] = make_float4(velocity, velocityFloat4.w);
	}

	static __device__ int3 CalculateGridPosition(float4 position)
	{
		// Convert a world space position into grid coordinates
		int3 gridPosition;
		float3 gridPositionFloat3 = (make_float3(position) - c_parameters.worldMin) / c_parameters.cellSize;
		gridPosition.x = floor(gridPositionFloat3.x);
		gridPosition.y = floor(gridPositionFloat3.y);
		gridPosition.z = floor(gridPositionFloat3.z);
		return gridPosition;
	}

	static __device__ unsigned int CalculateGridHash(int3 gridPosition)
	{
		// Use the particles position and the grid size to calculate a basic and universal hash
		return __mul24(gridPosition.z, c_parameters.gridSizeYX) + __mul24(gridPosition.y, c_parameters.gridSize.x) + gridPosition.x;
	}

	static __global__ void CalculateHashKernel(float4* position, uint2* particleHash)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Calculate the grid position of the particle.
		int3 gridPosition = CalculateGridPosition(position[index]);

		// Calculate the grid hash of the particle
		unsigned int gridHash = CalculateGridHash(gridPosition);

		// Use the calculated hash to create a key value pair containing the position index
		particleHash[index] = make_uint2(gridHash, index);
	}

	static __global__ void ReorderKernel(uint2* particleHash, unsigned int* cellStart, float4* oldPosition, float4* oldVelocity, float4* sortedPosition, float4* sortedVelocity)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Load hash value of the current particle into shared memory
		uint2 sortedIndex = particleHash[index];
		__shared__ unsigned int sharedHash[257];

		// Account for the previous particle's hash
		sharedHash[threadIdx.x + 1] = sortedIndex.x;
		// If the current particle is not the first particle in the block, then load the hash value of the previous particle into shared memory
		if (index > 0 && threadIdx.x == 0)
		{
			volatile uint2 prevData = particleHash[index - 1];
			sharedHash[0] = prevData.x;
		}

		__syncthreads();

		// If the hash values are not equal, then load the index of the current particle into the cellStart array.
		if (index == 0 || sortedIndex.x != sharedHash[threadIdx.x]) {
			cellStart[sortedIndex.x] = index;
		}

		// Use sorted index as index to textures holding position and velocity data
		sortedPosition[index] = tex1Dfetch(oldPositionTexture, sortedIndex.y);
		sortedVelocity[index] = tex1Dfetch(oldVelocityTexture, sortedIndex.y);
	}

	static __device__ float CalculateCellDensity(int3 gridPosition, unsigned int index, float4 position, float4* oldPosition, uint2* particleHash, unsigned int* cellStart)
	{
		float density = 0.0f;

		// Calculate the grid hash for the current particle
		unsigned int gridHash = CalculateGridHash(gridPosition);
		// Fetch the start index of the cell in the cellStartTexture.
		unsigned int bucketStart = tex1Dfetch(cellStartTexture, gridHash);
		// If the start index is 0xffffffff, then the cell is empty and the density is set to 0.
		if (bucketStart == 0xffffffff) {
			return density;
		}

		for (unsigned int i = 0; i < c_parameters.maxParticlesInCellCount; i++)
		{
			unsigned int indexOther = bucketStart + i;
			// Fetch the hash value of the current cell from the particleHashTexture.
			uint2 sortedIndex = tex1Dfetch(particleHashTexture, indexOther);

			if (sortedIndex.x != gridHash) {
				break;
			}

			if (indexOther != index)
			{
				// Calculate the relative position between the current particle and the other particle.
				float4 positionOther = tex1Dfetch(oldPositionTexture, indexOther);

				float4 p = position - positionOther;
				float r2 = p.x * p.x + p.y * p.y + p.z * p.z;

				if (r2 < c_parameters.smoothingRadius)
				{
					float c = c_parameters.smoothingRadius - r2;
					density += c * c * c;
				}
			}
		}

		return density;
	}

	static __device__ float3 CalculatePairForce(float4 relativePosition, float4 relativeVelocity, float PPAdd, float PPMultiply)
	{
		// Calculate the distance between the two particles
		float3 relPos = *(float3*)&relativePosition.x;
		float3 relVel = *(float3*)&relativeVelocity.x;
		float r = max(c_parameters.minDist, length(relPos));

		// If the distance is less than the minimum distance, the force is set to zero
		float3 force = make_float3(0.0f);

		// If the distance is greater than the minimum distance, the force is calculated.
		if (r < c_parameters.homogenity)
		{
			// Scale the force by the distance between the two particles.
			float c = c_parameters.homogenity - r;
			float pterm = c * c_parameters.spikyKern * PPAdd / r;
			float vterm = c_parameters.lapKern * c_parameters.viscosity;

			force = pterm * relPos + vterm * relVel;
			force *= c * PPMultiply;
		}

		return force;
	}

	static __device__ float3 CalculateCellForce(int3 gridPosition, unsigned int index, float4 position, float4 velocity, float4* oldPosition, float4* oldVelocity, float currentPressure, float currentDensity, float* pressure, float* density, uint2* particleHash, unsigned int* cellStart)
	{
		float3 force = make_float3(0.0f);

		// Calculate the grid hash for the current particle.
		unsigned int gridHash = CalculateGridHash(gridPosition);
		// Fetch the start index of the cell from the cellStartTexture.
		unsigned int bucketStart = tex1Dfetch(cellStartTexture, gridHash);
		// If the start index is 0xffffffff, then the cell is empty and the force is set to 0.
		if (bucketStart == 0xffffffff) {
			return force;
		}

		for (unsigned int i = 0; i < c_parameters.maxParticlesInCellCount; i++)
		{
			unsigned int indexOther = bucketStart + i;
			// Fetch the hash value of the current cell from the particleHashTexture.
			uint2 sortedIndex = tex1Dfetch(particleHashTexture, indexOther);

			if (sortedIndex.x != gridHash) {
				break;
			}

			if (indexOther != index)
			{
				// Fetch the position, velocity, pressure and density of the current cell from the oldPositionTexture, oldVelocityTexture, pressureTexture and densityTexture respectively
				float4 positionOther = tex1Dfetch(oldPositionTexture, indexOther);
				float4 velocityOther = tex1Dfetch(oldVelocityTexture, indexOther);
				float pressureOther = tex1Dfetch(pressureTexture, indexOther);
				float densityOther = tex1Dfetch(densityTexture, indexOther);

				// If the density of the current cell is less than the minimum density, the density of the current cell is set to the minimum density.
				float d12 = min(c_parameters.minDens, 1.0f / (currentDensity * densityOther));
				// Calculate the pair force between the current and the other cell
				force += CalculatePairForce(position - positionOther, velocityOther - velocity, currentPressure + pressureOther, d12);
			}
		}

		return force;
	}

	static __global__ void CalculateDensityKernel(float4* oldPosition, float* pressure, float* density, uint2* particleHash, unsigned int* cellStart)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		// Fetch the position of the particle from the oldPositionTexture.
		float4 position = tex1Dfetch(oldPositionTexture, index);
		// Calculate the grid position of the particle.
		int3 gridPos = CalculateGridPosition(position);

		float sum = 0.0f;

		// Calculate the density of the particle
		const int s = 1;
		for (int z = -s; z <= s; z++) {
			for (int y = -s; y <= s; y++) {
				for (int x = -s; x <= s; x++) {
					sum += CalculateCellDensity(gridPos + make_int3(x, y, z), index, position, oldPosition, particleHash, cellStart);
				}
			}
		}

		// Use common forumlae to calculate density and pressure values
		float newDensity = sum * c_parameters.poly6Kern * c_parameters.particleMass;
		float newPressure = (newDensity - c_parameters.restDensity) * c_parameters.stiffness;

		// Store the new values
		pressure[index] = newPressure;
		density[index] = newDensity;
	}

	static __global__ void CalculateForceKernel(float4* newPosition, float4* newVelocity, float4* oldPosition, float4* oldVelocity, float* pressure, float* density, uint2* particleHash, unsigned int* cellStart)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Fetch the position, velocity, pressure and density of the current cell from the oldPositionTexture, oldVelocityTexture, pressureTexture and densityTexture respectively
		float4 position = tex1Dfetch(oldPositionTexture, index);
		float4 currentVelocity = tex1Dfetch(oldVelocityTexture, index);
		float currentPressure = tex1Dfetch(pressureTexture, index);
		float currentDensity = tex1Dfetch(densityTexture, index);

		// Calculate the grid position of the particle
		int3 gridPos = CalculateGridPosition(position);

		// Calculate the force that is being exerted onto the particle
		float3 velocity = make_float3(0.0f);
		const int s = 1;
		for (int z = -s; z <= s; z++) {
			for (int y = -s; y <= s; y++) {
				for (int x = -s; x <= s; x++) {
					velocity += CalculateCellForce(gridPos + make_int3(x, y, z), index, position, currentVelocity, oldPosition, oldVelocity,
						currentPressure, currentDensity, pressure, density, particleHash, cellStart);
				}
			}
		}

		volatile unsigned int si = particleHash[index].y;
		velocity *= c_parameters.particleMass * c_parameters.timeStep;

		// Store the new value
		newVelocity[si] = currentVelocity + make_float4(velocity, 0.0f);
	}
}

#endif // !SIMULATION_KERNEL_CU_
