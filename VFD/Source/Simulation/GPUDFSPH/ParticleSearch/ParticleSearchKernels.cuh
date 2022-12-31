#ifndef PARTICLE_SEARCH_KERNELS_CUH
#define PARTICLE_SEARCH_KERNELS_CUH

#include "SearchInfo.h"
#include "Simulation/GPUDFSPH/DFSPHParticle.h"

#define CUDA_MAX_NEIGHBORS 70
#define CUDA_META_GRID_GROUP_SIZE 8
#define CUDA_META_GRID_BLOCK_SIZE (CUDA_META_GRID_GROUP_SIZE * CUDA_META_GRID_GROUP_SIZE * CUDA_META_GRID_GROUP_SIZE)

__global__ void ComputeMinMaxKernel(
	const vfd::DFSPHParticle* points,
	unsigned int pointCount,
	float searchRadius,
	glm::ivec3* minCell,
	glm::ivec3* maxCell
);

__global__ void InsertParticlesMortonKernel(
	vfd::SearchInfo info,
	const vfd::DFSPHParticle* points,
	unsigned int* pointCellIndices,
	unsigned int* cellpointCounts,
	unsigned int* sortIndices
);

__global__ void CountingSortIndicesKernel(
	vfd::SearchInfo info,
	const unsigned int* pointCellIndices,
	const unsigned int* cellOffsets,
	const unsigned int* sortIndicesSrc,
	unsigned int* sortIndicesDest
);

__global__ void ComputeCountsKernel(
	const vfd::DFSPHParticle* points,
	vfd::SearchInfo info,
	const unsigned int* cellOffsets,
	const unsigned int* cellPointCounts,
	unsigned int* neighborCounts,
	const unsigned int* reversedSortIndices
);

__global__ void NeighborhoodQueryWithCountsKernel(
	const vfd::DFSPHParticle* points,
	vfd::SearchInfo info,
	const unsigned int* cellOffsets,
	const unsigned int* cellPointCounts,
	const unsigned int* neighborWriteOffsets,
	unsigned int* neighbors,
	const unsigned int* reversedSortIndices
);

__global__ void PointSortKernel(
	vfd::DFSPHParticle* data,
	vfd::DFSPHParticle* copy,
	unsigned int* sortedIndices,
	unsigned int pointCount
);

#endif // !PARTICLE_SEARCH_KERNELS_CUH