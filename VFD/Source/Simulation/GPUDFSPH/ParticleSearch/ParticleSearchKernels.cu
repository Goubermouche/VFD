#include "pch.h"
#include "ParticleSearchKernels.cuh"

__device__ __forceinline__ inline unsigned int CellIndicesToLinearIndex(const glm::uvec3& cellDimensions, glm::ivec3& xyz)
{
	return xyz.z * cellDimensions.y * cellDimensions.x + xyz.y * cellDimensions.x + xyz.x;
}

__device__ __forceinline__ inline unsigned int Part1By2(unsigned int x)
{
	x &= 0x000003ff;                // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ x << 16) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ x << 8 ) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ x << 4 ) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ x << 2 ) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

__device__ __forceinline__ inline unsigned int MortonCode3(unsigned int x, unsigned int y, unsigned int z)
{
	return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

__device__ __forceinline__ unsigned int ToCellIndexMortonMetaGrid(const vfd::SearchInfo& info, glm::ivec3 gridCell)
{
	glm::ivec3 metaGridCell = {
		gridCell.x / CUDA_META_GRID_GROUP_SIZE,
		gridCell.y / CUDA_META_GRID_GROUP_SIZE,
		gridCell.z / CUDA_META_GRID_GROUP_SIZE
	};

	gridCell.x %= CUDA_META_GRID_GROUP_SIZE;
	gridCell.y %= CUDA_META_GRID_GROUP_SIZE;
	gridCell.z %= CUDA_META_GRID_GROUP_SIZE;

	const unsigned int metaGridIndex = CellIndicesToLinearIndex(info.MetaGridDimension, metaGridCell);
	return metaGridIndex * CUDA_META_GRID_BLOCK_SIZE + MortonCode3(gridCell.x, gridCell.y, gridCell.z);
}

__global__ void ComputeMinMaxKernel(const vfd::DFSPHParticle* points, unsigned int pointCount, float searchRadius, glm::ivec3* minCell, glm::ivec3* maxCell)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= pointCount) {
		return;
	}

	const glm::vec3 particle = points[particleIndex].Position;

	glm::ivec3 cell;
	cell.x = static_cast<int>(floor(particle.x / searchRadius));
	cell.y = static_cast<int>(floor(particle.y / searchRadius));
	cell.z = static_cast<int>(floor(particle.z / searchRadius));

	atomicMin(&(minCell->x), cell.x);
	atomicMin(&(minCell->y), cell.y);
	atomicMin(&(minCell->z), cell.z);

	atomicMax(&(maxCell->x), cell.x);
	atomicMax(&(maxCell->y), cell.y);
	atomicMax(&(maxCell->z), cell.z);
}

__global__ void InsertParticlesMortonKernel(vfd::SearchInfo info, const vfd::DFSPHParticle* points, unsigned int* pointCellIndices, unsigned int* cellpointCounts, unsigned int* sortIndices)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= info.ParticleCount) {
		return;
	}

	const glm::vec3 gridCellF = (points[particleIndex].Position - info.GridMin) * info.GridDelta;
	const glm::ivec3 gridCell = static_cast<glm::ivec3>(gridCellF);

	unsigned int cellIndex = ToCellIndexMortonMetaGrid(info, gridCell);
	pointCellIndices[particleIndex] = cellIndex;
	sortIndices[particleIndex] = atomicAdd(&cellpointCounts[cellIndex], 1);
}

__global__ void CountingSortIndicesKernel(vfd::SearchInfo info, const unsigned int* pointCellIndices, const unsigned int* cellOffsets, const unsigned int* sortIndicesSrc, unsigned int* sortIndicesDest)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= info.ParticleCount) {
		return;
	}

	unsigned int gridCellIndex = pointCellIndices[particleIndex];
	unsigned int sortIndex = sortIndicesSrc[particleIndex] + cellOffsets[gridCellIndex];
	sortIndicesDest[sortIndex] = particleIndex;
}

__global__ void ComputeCountsKernel(const vfd::DFSPHParticle* points, vfd::SearchInfo info, const unsigned int* cellOffsets, const unsigned int* cellPointCounts, unsigned int* neighborCounts, const unsigned int* reversedSortIndices)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= info.ParticleCount) {
		return;
	}

	const glm::vec3 particle = points[particleIndex].Position;
	const glm::vec3 gridCellF = (particle - info.GridMin) * info.GridDelta;
	const glm::ivec3 gridCell = { static_cast<int>(floor(gridCellF.x)), static_cast<int>(floor(gridCellF.y)), static_cast<int>(floor(gridCellF.z)) };
	unsigned int neighborCount = 0;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++)
			{
				glm::ivec3 finalCoord = gridCell + glm::ivec3(x, y, z);

				if (finalCoord.x < 0 || finalCoord.y < 0 || finalCoord.z < 0 || finalCoord.x >= info.GridDimension.x || finalCoord.y >= info.GridDimension.y || finalCoord.z >= info.GridDimension.z) {
					continue;
				}

				unsigned int neighborCellIndex = ToCellIndexMortonMetaGrid(info, finalCoord);
				unsigned int neighborCellCount = cellPointCounts[neighborCellIndex];
				unsigned int neighborCellStart = cellOffsets[neighborCellIndex];

				for (unsigned int i = neighborCellStart; i < neighborCellStart + neighborCellCount; i++)
				{
					unsigned int& neighborIndex = i;
					glm::vec3 diff = points[reversedSortIndices[neighborIndex]].Position - particle;
					float squaredDistance = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

					if (squaredDistance < info.SquaredSearchRadius && squaredDistance > 0.0f)
					{
						neighborCount++;
					}

					if (neighborCount == CUDA_MAX_NEIGHBORS)
					{
						neighborCounts[particleIndex] = neighborCount;
						return;
					}
				}
			}
		}
	}

	neighborCounts[particleIndex] = neighborCount;
}

__global__ void NeighborhoodQueryWithCountsKernel(const vfd::DFSPHParticle* points, vfd::SearchInfo info, const unsigned int* cellOffsets, const unsigned int* cellPointCounts, const unsigned int* neighborWriteOffsets, unsigned int* neighbors, const unsigned int* reversedSortIndices)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= info.ParticleCount) {
		return;
	}

	const glm::vec3 particle = points[particleIndex].Position;
	const glm::vec3 gridCellF = (particle - info.GridMin) * info.GridDelta;
	const glm::ivec3 gridCell = { static_cast<int>(floor(gridCellF.x)), static_cast<int>(floor(gridCellF.y)), static_cast<int>(floor(gridCellF.z)) };

	unsigned int neighborCount = 0;
	const unsigned int writeOffset = neighborWriteOffsets[particleIndex];

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++)
			{
				glm::ivec3 finalCoord = gridCell + glm::ivec3(x, y, z);

				if (finalCoord.x < 0 || finalCoord.y < 0 || finalCoord.z < 0 || finalCoord.x >= info.GridDimension.x || finalCoord.y >= info.GridDimension.y || finalCoord.z >= info.GridDimension.z) {
					continue;
				}

				unsigned int neighborCellIndex = ToCellIndexMortonMetaGrid(info, finalCoord);
				unsigned int neighborCellCount = cellPointCounts[neighborCellIndex];
				unsigned int neighborCellStart = cellOffsets[neighborCellIndex];

				for (unsigned int i = neighborCellStart; i < neighborCellStart + neighborCellCount; i++)
				{
					unsigned int& neighborIndex = i;
					glm::vec3 diff = points[reversedSortIndices[neighborIndex]].Position - particle;
					float squaredDistance = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

					if (squaredDistance < info.SquaredSearchRadius && squaredDistance > 0.0f)
					{
						neighbors[writeOffset + neighborCount] = reversedSortIndices[neighborIndex];
						neighborCount++;
					}

					if (neighborCount == CUDA_MAX_NEIGHBORS)
					{
						return;
					}
				}
			}
		}
	}
}

__global__ void PointSortKernel(vfd::DFSPHParticle* data, vfd::DFSPHParticle* copy, unsigned int* sortedIndices, unsigned int pointCount)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= pointCount)
	{
		return;
	}

	// data[i] = copy[sortedIndices[i]];

	data[i].Position = copy[sortedIndices[i]].Position;
	data[i].Velocity = copy[sortedIndices[i]].Velocity;
	data[i].Acceleration = copy[sortedIndices[i]].Acceleration;
	data[i].Mass = copy[sortedIndices[i]].Mass;
	data[i].Density = copy[sortedIndices[i]].Density;
	// data[i].Kappa = copy[sortedIndices[i]].Kappa;
	// data[i].KappaVelocity = copy[sortedIndices[i]].KappaVelocity;
	// data[i].Position = copy[sortedIndices[i]].Position;
	data[i].PressureRho2 = copy[sortedIndices[i]].PressureRho2;
}
