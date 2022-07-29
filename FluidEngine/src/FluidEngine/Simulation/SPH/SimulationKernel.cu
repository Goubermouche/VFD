#ifndef SIMULATION_KERNEL_CU_
#define SIMULATION_KERNEL_CU_

#include "FluidEngine/Compute/Utility/CUDA/cutil_math.h"
#include "SimulationParameters.cuh"

namespace fe {


	__constant__ SimulationData c_Description;

	static __device__ void CalculateBoundary(glm::vec3& position, glm::vec3& velocity)
	{
		glm::vec3 worldMin = c_Description.worldMin;
		glm::vec3 worldMax = c_Description.worldMax;
		glm::vec3 normal;

		float bounds = c_Description.boundsSoftDistance;
		float stiffness = c_Description.boundsStiffness;
		float damping0 = c_Description.boundsDamping;
		float damping1 = c_Description.boundsDampingCritical;
		float acceleration;
		float difference;

#define  EPS	0.00001f // epsilon 
#define  ADD_BOUNDS0()  acceleration = stiffness * difference - damping0 * glm::dot(normal, velocity);  velocity += acceleration * normal * c_Description.timeStep;
#define  ADD_BOUNDS1()  acceleration = stiffness * difference - damping1 * glm::dot(normal, velocity);  velocity += acceleration * normal * c_Description.timeStep;

		// Box bounds
		difference = bounds - position.z + worldMin.z;
		if (difference > EPS) { normal = glm::vec3(0, 0, 1); ADD_BOUNDS1(); }
		difference = bounds + position.z - worldMax.z;
		if (difference > EPS) { normal = glm::vec3(0, 0, -1); ADD_BOUNDS1(); }
		difference = bounds - position.x + worldMin.x;
		if (difference > EPS) { normal = glm::vec3(1, 0, 0); ADD_BOUNDS0(); }
		difference = bounds + position.x - worldMax.x;
		if (difference > EPS) { normal = glm::vec3(-1, 0, 0); ADD_BOUNDS0(); }
		difference = bounds - position.y + worldMin.y;
		if (difference > EPS) { normal = glm::vec3(0, 1, 0); ADD_BOUNDS0(); }
		difference = bounds + position.y - worldMax.y;
		if (difference > EPS) { normal = glm::vec3(0, -1, 0); ADD_BOUNDS0(); }
	}

	static __global__ void IntegrateKernel(glm::vec3* newPosition, glm::vec3* oldPosition, glm::vec3* newVelocity, glm::vec3* oldVelocity) {
		uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Load position and velocity into registers
		glm::vec3 position = oldPosition[index];
		glm::vec3 velocity = oldVelocity[index];

		// Calculate boundary conditions
		CalculateBoundary(position, velocity);

		// Add gravity force to velocity
		velocity += c_Description.gravity * c_Description.timeStep;
		velocity *= c_Description.globalDamping;

		// Update the position of the particle
		position += velocity * c_Description.timeStep;

		// Clamp the position to the world boundaries
		float b = c_Description.boundsHardDistance;
		glm::vec3 wmin = c_Description.worldMin, wmax = c_Description.worldMax;
		if (position.x > wmax.x - b) { position.x = wmax.x - b; }
		if (position.x < wmin.x + b) { position.x = wmin.x + b; }
		if (position.y > wmax.y - b) { position.y = wmax.y - b; }
		if (position.y < wmin.y + b) { position.y = wmin.y + b; }
		if (position.z > wmax.z - b) { position.z = wmax.z - b; }
		if (position.z < wmin.z + b) { position.z = wmin.z + b; }

		// Stores the new position and velocity of the particle
		newPosition[index] = position;
		newVelocity[index] = velocity;
	}

	static __device__ int3 CalculateGridPosition(glm::vec3 position)
	{
		// Convert a world space position into grid coordinates
		int3 gridPosition;
		glm::vec3 gridPositionFloat3 = (position - c_Description.worldMin) / c_Description.cellSize;
		gridPosition.x = floor(gridPositionFloat3.x);
		gridPosition.y = floor(gridPositionFloat3.y);
		gridPosition.z = floor(gridPositionFloat3.z);
		return gridPosition;
	}

	static __device__ uint32_t CalculateGridHash(int3 gridPosition)
	{
		// Use the particles position and the grid size to calculate a basic and universal hash
		return __mul24(gridPosition.z, c_Description.gridSizeYX) + __mul24(gridPosition.y, c_Description.gridSize.x) + gridPosition.x;
	}

	static __global__ void CalculateHashKernel(glm::vec3* position, glm::uvec2* particleHash)
	{
		uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Calculate the grid position of the particle.
		int3 gridPosition = CalculateGridPosition(position[index]);

		// Calculate the grid hash of the particle
		uint32_t gridHash = CalculateGridHash(gridPosition);

		// Use the calculated hash to create a key value pair containing the position index
		particleHash[index] = glm::uvec2(gridHash, index);
	}

	static __global__ void ReorderKernel(glm::uvec2* particleHash, uint32_t* cellStart, glm::vec3* oldPosition, glm::vec3* oldVelocity, glm::vec3* sortedPosition, glm::vec3* sortedVelocity)
	{
		uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Load hash value of the current particle into shared memory
		glm::uvec2 sortedIndex = particleHash[index];
		__shared__ uint32_t sharedHash[257];

		// Account for the previous particle's hash
		sharedHash[threadIdx.x + 1] = sortedIndex.x;
		// If the current particle is not the first particle in the block, then load the hash value of the previous particle into shared memory
		if (index > 0 && threadIdx.x == 0)
		{
			volatile glm::uvec2 prevData = particleHash[index - 1];
			sharedHash[0] = prevData.x;
		}

		__syncthreads();

		// If the hash values are not equal, then load the index of the current particle into the cellStart array.
		if (index == 0 || sortedIndex.x != sharedHash[threadIdx.x]) {
			cellStart[sortedIndex.x] = index;
		}

		// Use sorted index as index to textures holding position and velocity data
		sortedPosition[index] = oldPosition[sortedIndex.y];
		sortedVelocity[index] = oldVelocity[sortedIndex.y];
	}

	static __device__ float CalculateCellDensity(int3 gridPosition, uint32_t index, glm::vec3 position, glm::vec3* oldPosition, glm::uvec2* particleHash, uint32_t* cellStart)
	{
		float density = 0.0f;

		// Calculate the grid hash for the current particle
		uint32_t gridHash = CalculateGridHash(gridPosition);
		// Fetch the start index of the cell in the cellStartTexture.
		uint32_t bucketStart = cellStart[gridHash];
		// If the start index is 0xffffffff, then the cell is empty and the density is set to 0.
		if (bucketStart == 0xffffffff) {
			return density;
		}

		for (uint16_t i = 0; i < c_Description.maxParticlesInCellCount; i++)
		{
			uint32_t indexOther = bucketStart + i;
			// Fetch the hash value of the current cell from the particleHashTexture.
			glm::uvec2 sortedIndex = particleHash[indexOther];

			if (sortedIndex.x != gridHash) {
				break;
			}

			if (indexOther != index)
			{
				// Calculate the relative position between the current particle and the other particle.
				glm::vec3 positionOther = oldPosition[indexOther];

				glm::vec3 p = position - positionOther;
				float r2 = p.x * p.x + p.y * p.y + p.z * p.z;

				if (r2 < c_Description.smoothingRadius)
				{
					float c = c_Description.smoothingRadius - r2;
					density += c * c * c;
				}
			}
		}

		return density;
	}

	static __device__ glm::vec3 CalculatePairForce(glm::vec3 relativePosition, glm::vec3 relativeVelocity, float PPAdd, float PPMultiply)
	{
		// Calculate the distance between the two particles
		float r = max(c_Description.minDist, length(relativePosition));

		// If the distance is less than the minimum distance, the force is set to zero
		glm::vec3 force = glm::vec3(0, 0, 0);

		// If the distance is greater than the minimum distance, the force is calculated.
		if (r < c_Description.homogenity)
		{
			// Scale the force by the distance between the two particles.
			float c = c_Description.homogenity - r;
			float pterm = c * c_Description.spikyKern * PPAdd / r;
			float vterm = c_Description.lapKern * c_Description.viscosity;

			force = pterm * relativePosition + vterm * relativeVelocity;
			force *= c * PPMultiply;
		}

		return force;
	}

	static __device__ glm::vec3 CalculateCellForce(int3 gridPosition, uint32_t index, glm::vec3 position, glm::vec3 velocity, glm::vec3* oldPosition, glm::vec3* oldVelocity, float currentPressure, float currentDensity, float* pressure, float* density, glm::uvec2* particleHash, uint32_t* cellStart)
	{
		glm::vec3 force = glm::vec3(0, 0, 0);

		// Calculate the grid hash for the current particle.
		uint32_t gridHash = CalculateGridHash(gridPosition);
		// Fetch the start index of the cell from the cellStartTexture.
		uint32_t bucketStart = cellStart[gridHash];
		// If the start index is 0xffffffff, then the cell is empty and the force is set to 0.
		if (bucketStart == 0xffffffff) {
			return force;
		}

		for (uint16_t i = 0; i < c_Description.maxParticlesInCellCount; i++)
		{
			uint32_t indexOther = bucketStart + i;
			// Fetch the hash value of the current cell from the particleHashTexture.
			glm::uvec2 sortedIndex = particleHash[indexOther];

			if (sortedIndex.x != gridHash) {
				break;
			}

			if (indexOther != index)
			{
				// Fetch the position, velocity, pressure and density of the current cell from the oldPositionTexture, oldVelocityTexture, pressureTexture and densityTexture respectively
				glm::vec3 positionOther = oldPosition[indexOther];
				glm::vec3 velocityOther = oldVelocity[indexOther];
				float pressureOther = pressure[indexOther];
				float densityOther = density[indexOther];

				// If the density of the current cell is less than the minimum density, the density of the current cell is set to the minimum density.
				float d12 = min(c_Description.minDens, 1.0f / (currentDensity * densityOther));
				// Calculate the pair force between the current and the other cell
				force += CalculatePairForce(position - positionOther, velocityOther - velocity, currentPressure + pressureOther, d12);
			}
		}

		return force;
	}

	static __global__ void CalculateDensityKernel(glm::vec3* oldPosition, float* pressure, float* density, glm::uvec2* particleHash, uint32_t* cellStart)
	{
		// Fetch the position of the particle from the oldPositionTexture.
		uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		glm::vec3 position = oldPosition[index];
		// Calculate the grid position of the particle.
		int3 gridPos = CalculateGridPosition(position);

		float sum = 0.0f;

		// Calculate the density of the particle
		const int16_t s = 1;
		for (int16_t z = -s; z <= s; z++) {
			for (int16_t y = -s; y <= s; y++) {
				for (int16_t x = -s; x <= s; x++) {
					sum += CalculateCellDensity(gridPos + make_int3(x, y, z), index, position, oldPosition, particleHash, cellStart);
				}
			}
		}

		// Use common forumlae to calculate density and pressure values
		float newDensity = sum * c_Description.poly6Kern * c_Description.particleMass;
		float newPressure = (newDensity - c_Description.restDensity) * c_Description.stiffness;

		// Store the new values
		pressure[index] = newPressure;
		density[index] = newDensity;
	}

	static __global__ void CalculateForceKernel(glm::vec3* newPosition, glm::vec3* newVelocity, glm::vec3* oldPosition, glm::vec3* oldVelocity, float* pressure, float* density, glm::uvec2* particleHash, uint32_t* cellStart)
	{
		int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		// Fetch the position, velocity, pressure and density of the current cell from the oldPositionTexture, oldVelocityTexture, pressureTexture and densityTexture respectively
		glm::vec3 position = oldPosition[index];
		glm::vec3 currentVelocity = oldVelocity[index];
		float currentPressure = pressure[index];
		float currentDensity = density[index];

		// Calculate the grid position of the particle
		int3 gridPos = CalculateGridPosition(position);

		// Calculate the force that is being exerted onto the particle
		glm::vec3 velocity = glm::vec3(0, 0, 0);
		const int16_t s = 1;
		for (int16_t z = -s; z <= s; z++) {
			for (int16_t y = -s; y <= s; y++) {
				for (int16_t x = -s; x <= s; x++) {
					velocity += CalculateCellForce(gridPos + make_int3(x, y, z), index, position, currentVelocity, oldPosition, oldVelocity,
						currentPressure, currentDensity, pressure, density, particleHash, cellStart);
				}
			}
		}

		volatile uint32_t si = particleHash[index].y;
		velocity *= c_Description.particleMass * c_Description.timeStep;

		// Store the new value
		newVelocity[si] = currentVelocity + velocity;
	}
}

#endif // !SIMULATION_KERNEL_CU_