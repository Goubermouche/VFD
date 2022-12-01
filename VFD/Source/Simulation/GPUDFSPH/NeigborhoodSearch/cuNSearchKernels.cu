#include "cuNSearchKernels.cuh"

#define INT16_RANGE 32767
#define UPDATE_REF_OFFSET -32768

__host__ __device__ inline unsigned int Part1By2(unsigned int x)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x << 8)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x << 4)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x << 2)) & 0x09249249;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

__host__ __device__ inline unsigned int MortonCode3(unsigned int x, unsigned int y, unsigned int z)
{
	return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

__host__ __device__ inline unsigned int CellIndicesToLinearIndex(const glm::uvec3& cellDimensions, glm::ivec3& xyz)
{
	return xyz.z * cellDimensions.y * cellDimensions.x + xyz.y * cellDimensions.x + xyz.x;
}

inline __device__ unsigned int CellIndexToMortonMetaGrid(const GridInfo& GridInfo, glm::ivec3 gridCell)
{
	glm::ivec3 metaGridCell = glm::ivec3(
		gridCell.x / CUDA_META_GRID_GROUP_SIZE,
		gridCell.y / CUDA_META_GRID_GROUP_SIZE,
		gridCell.z / CUDA_META_GRID_GROUP_SIZE);

	gridCell.x %= CUDA_META_GRID_GROUP_SIZE;
	gridCell.y %= CUDA_META_GRID_GROUP_SIZE;
	gridCell.z %= CUDA_META_GRID_GROUP_SIZE;
	unsigned int metaGridIndex = CellIndicesToLinearIndex(GridInfo.MetaGridDimension, metaGridCell);

	return metaGridIndex * CUDA_META_GRID_BLOCK_SIZE + MortonCode3(gridCell.x, gridCell.y, gridCell.z);
}

__global__ void ComputeMinMaxKernel(const glm::vec3* particles, unsigned int particleCount, float m_SearchRadius, glm::ivec3* minCell, glm::ivec3* maxCell)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= particleCount) {
		return;
	}

	const glm::vec3 particle = particles[particleIndex];
	glm::ivec3 cell;

	cell.x = (int)floor(particle.x / m_SearchRadius);
	cell.y = (int)floor(particle.y / m_SearchRadius);
	cell.z = (int)floor(particle.z / m_SearchRadius);

	atomicMin(&(minCell->x), cell.x);
	atomicMin(&(minCell->y), cell.y);
	atomicMin(&(minCell->z), cell.z);

	atomicMax(&(maxCell->x), cell.x);
	atomicMax(&(maxCell->y), cell.y);
	atomicMax(&(maxCell->z), cell.z);
}

__global__ void InsertParticlesMortonKernel(const GridInfo GridInfo, const glm::vec3* particles, unsigned int* particleCellIndices, unsigned int* cellParticleCounts, unsigned int* sortIndices)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= GridInfo.ParticleCount)
	{
		return;
	}

	glm::vec3 gridCellF = (particles[particleIndex] - GridInfo.GridMin) * GridInfo.GridDelta;
	glm::ivec3 gridCell = glm::ivec3(int(gridCellF.x), int(gridCellF.y), int(gridCellF.z));
	unsigned int cellIndex = CellIndexToMortonMetaGrid(GridInfo, gridCell);
	particleCellIndices[particleIndex] = cellIndex;
	sortIndices[particleIndex] = atomicAdd(&cellParticleCounts[cellIndex], 1);
}

__global__ void CountingSortIndicesKernel(const GridInfo GridInfo, const unsigned int* particleCellIndices, const unsigned int* cellOffsets, const unsigned int* sortIndicesSrc, unsigned int* sortIndicesDest)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= GridInfo.ParticleCount)
	{
		return;
	}

	unsigned int gridCellIndex = particleCellIndices[particleIndex];
	unsigned int sortIndex = sortIndicesSrc[particleIndex] + cellOffsets[gridCellIndex];
	sortIndicesDest[sortIndex] = particleIndex;
}

__global__ void ComputeCountsKernel(const glm::vec3* queryPoints, const unsigned int queryPointCount, const GridInfo GridInfo, const glm::vec3* particles, const unsigned int* cellOffsets, const unsigned int* cellParticleCounts, unsigned int* neighborCounts, const unsigned int* reversedSortIndices)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= queryPointCount)
	{
		return;
	}

	const glm::vec3 particle = queryPoints[particleIndex];
	glm::vec3 gridCellF = (particle - GridInfo.GridMin) * GridInfo.GridDelta;
	glm::ivec3 coord = glm::ivec3(int(floor(gridCellF.x)), int(floor(gridCellF.y)), int(floor(gridCellF.z)));
	unsigned int neighborCount = 0;

	for (int z = -1; z < 2; z++)
	{
		for (int y = -1; y < 2; y++)
		{
			for (int x = -1; x < 2; x++)
			{
				glm::ivec3 finalCoord = coord + glm::ivec3(x, y, z);

				if (finalCoord.x < 0 || finalCoord.y < 0 || finalCoord.z < 0
					|| finalCoord.x >= GridInfo.GridDimension.x || finalCoord.y >= GridInfo.GridDimension.y || finalCoord.z >= GridInfo.GridDimension.z)
				{
					continue;
				}

				unsigned int neighborCellIndex = CellIndexToMortonMetaGrid(GridInfo, finalCoord);
				unsigned int neighborCellCount = cellParticleCounts[neighborCellIndex];
				unsigned int neighborCellStart = cellOffsets[neighborCellIndex];

				for (unsigned int i = neighborCellStart; i < neighborCellStart + neighborCellCount; i++)
				{
					unsigned int& neighborIndex = i;
					glm::vec3 diff = particles[reversedSortIndices[neighborIndex]] - particle;
					float squaredDistance = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

					if (squaredDistance < GridInfo.SquaredSearchRadius && squaredDistance > 0.0)
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

__global__ void NeighborhoodQueryWithCountsKernel(const glm::vec3* queryPoints, const unsigned int queryPointCount, const GridInfo GridInfo, const glm::vec3* particles, const unsigned int* cellOffsets, const unsigned int* cellParticleCounts, const unsigned int* neighborWriteOffsets, unsigned int* neighbors, const unsigned int* reversedSortIndices)
{
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= queryPointCount)
	{
		return;
	}

	const glm::vec3 particle = queryPoints[particleIndex];
	glm::vec3 gridCellF = (particle - GridInfo.GridMin) * GridInfo.GridDelta;
	glm::ivec3 coord = glm::ivec3(int(floor(gridCellF.x)), int(floor(gridCellF.y)), int(floor(gridCellF.z)));
	unsigned int neighborCount = 0;
	const unsigned int writeOffset = neighborWriteOffsets[particleIndex];

	for (int z = -1; z < 2; z++)
	{
		for (int y = -1; y < 2; y++)
		{
			for (int x = -1; x < 2; x++)
			{
				glm::ivec3 finalCoord = coord + glm::ivec3(x, y, z);

				if (finalCoord.x < 0 || finalCoord.y < 0 || finalCoord.z < 0
					|| finalCoord.x >= GridInfo.GridDimension.x || finalCoord.y >= GridInfo.GridDimension.y || finalCoord.z >= GridInfo.GridDimension.z)
				{
					continue;
				}

				unsigned int neighborCellIndex = CellIndexToMortonMetaGrid(GridInfo, finalCoord);
				unsigned int neighborCellCount = cellParticleCounts[neighborCellIndex];
				unsigned int neighborCellStart = cellOffsets[neighborCellIndex];

				for (unsigned int i = neighborCellStart; i < neighborCellStart + neighborCellCount; i++)
				{
					unsigned int& neighborIndex = i;
					glm::vec3 diff = particles[reversedSortIndices[neighborIndex]] - particle;
					float squaredDistance = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

					if (squaredDistance < GridInfo.SquaredSearchRadius && squaredDistance > 0.0)
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
