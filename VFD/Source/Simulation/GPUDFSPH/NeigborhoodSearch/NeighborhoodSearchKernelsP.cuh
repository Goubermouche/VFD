#ifndef NEIGHBORHOOD_SEARCH_KERNELS_P_CUH
#define NEIGHBORHOOD_SEARCH_KERNELS_P_CUH

#include "GridInfoP.h"
#include "../DFSPHParticle.h"

#define CUDA_MAX_NEIGHBORS 70
#define CUDA_META_GRID_GROUP_SIZE 8
#define CUDA_META_GRID_BLOCK_SIZE (CUDA_META_GRID_GROUP_SIZE*CUDA_META_GRID_GROUP_SIZE*CUDA_META_GRID_GROUP_SIZE)

using namespace vfd;

__global__ void ComputeMinMaxKernelP(
	const DFSPHParticle* particles,
	unsigned int particleCount,
	float m_SearchRadius,
	glm::ivec3* minCell,
	glm::ivec3* maxCell
);

__global__ void InsertParticlesMortonKernelP(
	const GridInfo gridInfo,
	const DFSPHParticle* particles,
	unsigned int* particleCellIndices,
	unsigned int* cellParticleCounts,
	unsigned int* sortIndices
);

__global__ void CountingSortIndicesKernelP(
	const GridInfo gridInfo,
	const unsigned int* particleCellIndices,
	const unsigned int* cellOffsets,
	const unsigned int* sortIndicesSrc,
	unsigned int* sortIndicesDest
);

__global__ void ComputeCountsKernelP(
	const DFSPHParticle* queryPoints,
	const unsigned int queryPointCount,

	const GridInfo gridInfo,
	const DFSPHParticle* particles,
	const unsigned int* cellOffsets,
	const unsigned int* cellParticleCounts,
	unsigned int* neighborCounts,
	const unsigned int* reversedSortIndices
);

__global__ void NeighborhoodQueryWithCountsKernelP(
	const DFSPHParticle* queryPoints,
	const unsigned int queryPointCount,

	const GridInfo gridInfo,
	const DFSPHParticle* particles,
	const unsigned int* cellOffsets,
	const unsigned int* cellParticleCounts,
	const unsigned int* neighborWriteOffsets,
	unsigned int* neighbors,
	const unsigned int* reversedSortIndices
);

__global__ void SortKernelP(
	DFSPHParticle* data,
	DFSPHParticle* copy,
	unsigned int* sortedIndices,
	const unsigned int pointCount
);

#endif // !NEIGHBORHOOD_SEARCH_KERNELS_CUH