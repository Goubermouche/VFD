#ifndef CUN_SEARCH_KERNELS_CUH
#define CUN_SEARCH_KERNELS_CUH

#include <cuda_runtime.h>
#include "GridInfo.h"

#define CUDA_MAX_NEIGHBORS 70
#define CUDA_META_GRID_GROUP_SIZE 8
#define CUDA_META_GRID_BLOCK_SIZE (CUDA_META_GRID_GROUP_SIZE*CUDA_META_GRID_GROUP_SIZE*CUDA_META_GRID_GROUP_SIZE)

using namespace vfdcu;

__global__ void ComputeMinMaxKernel(
	const glm::vec3* particles,
	unsigned int particleCount,
	float m_SearchRadius,
	glm::ivec3* minCell,
	glm::ivec3* maxCell
);

__global__ void InsertParticlesMortonKernel(
	const GridInfo GridInfo,
	const glm::vec3* particles,
	unsigned int* particleCellIndices,
	unsigned int* cellParticleCounts,
	unsigned int* sortIndices
);

__global__ void CountingSortIndicesKernel(
	const GridInfo GridInfo,
	const unsigned int* particleCellIndices,
	const unsigned int* cellOffsets,
	const unsigned int* sortIndicesSrc,
	unsigned int* sortIndicesDest
);

__global__ void ComputeCountsKernel(
	const glm::vec3* queryPoints,
	const unsigned int queryPointCount,

	const GridInfo GridInfo,
	const glm::vec3* particles,
	const unsigned int* cellOffsets,
	const unsigned int* cellParticleCounts,
	unsigned int* neighborCounts,
	const unsigned int* reversedSortIndices
);

__global__ void NeighborhoodQueryWithCountsKernel(
	const glm::vec3* queryPoints,
	const unsigned int queryPointCount,

	const GridInfo GridInfo,
	const glm::vec3* particles,
	const unsigned int* cellOffsets,
	const unsigned int* cellParticleCounts,
	const unsigned int* neighborWriteOffsets,
	unsigned int* neighbors,
	const unsigned int* reversedSortIndices
);

#endif // !CUN_SEARCH_KERNELS_CUH