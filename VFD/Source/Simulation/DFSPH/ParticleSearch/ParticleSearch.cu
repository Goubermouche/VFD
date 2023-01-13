#include "pch.h"
#include "ParticleSearch.h"

#include "ParticleSearchKernels.cuh"

#include <thrust/sequence.h>
#include <thrust/gather.h>

namespace vfd
{
	void ParticleSearch::ComputeMinMax()
	{
		glm::ivec3 data[2];
		data[0] = glm::ivec3(std::numeric_limits<int>::max());
		data[1] = glm::ivec3(std::numeric_limits<int>::min());
		d_MinMax.resize(2);

		ComputeHelper::MemcpyHostToDevice(data, ComputeHelper::GetPointer(d_MinMax), 2);

		ComputeMinMaxKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			m_Particles,
			m_ParticleCount,
			m_SearchRadius,
			ComputeHelper::GetPointer(d_MinMax),
			ComputeHelper::GetPointer(d_MinMax) + 1
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		ComputeHelper::MemcpyDeviceToHost(ComputeHelper::GetPointer(d_MinMax), data, 2);

		const glm::ivec3 minCell = data[0];
		const glm::ivec3 maxCell = data[1] + glm::ivec3(1, 1, 1);

		m_Bounds.min = static_cast<glm::vec3>(minCell) * m_SearchRadius;
		m_Bounds.max = static_cast<glm::vec3>(maxCell) * m_SearchRadius;
	}

	void ParticleSearch::ComputeCellInformation()
	{
		m_GridInfo.ParticleCount = m_ParticleCount;
		m_GridInfo.SquaredSearchRadius = m_SearchRadius * m_SearchRadius;
		m_GridInfo.GridMin = m_Bounds.min;

		const float cellSize = m_SearchRadius;
		glm::vec3 gridSize = m_Bounds.Diagonal();
		m_GridInfo.GridDimension.x = static_cast<unsigned int>(ceil(gridSize.x / cellSize));
		m_GridInfo.GridDimension.y = static_cast<unsigned int>(ceil(gridSize.y / cellSize));
		m_GridInfo.GridDimension.z = static_cast<unsigned int>(ceil(gridSize.z / cellSize));

		m_GridInfo.GridDimension.x += 4;
		m_GridInfo.GridDimension.y += 4;
		m_GridInfo.GridDimension.z += 4;
		m_GridInfo.GridMin -= glm::vec3(cellSize, cellSize, cellSize) * 2.0f;

		m_GridInfo.MetaGridDimension.x = static_cast<unsigned int>(ceil(m_GridInfo.GridDimension.x / static_cast<float>(CUDA_META_GRID_GROUP_SIZE)));
		m_GridInfo.MetaGridDimension.y = static_cast<unsigned int>(ceil(m_GridInfo.GridDimension.y / static_cast<float>(CUDA_META_GRID_GROUP_SIZE)));
		m_GridInfo.MetaGridDimension.z = static_cast<unsigned int>(ceil(m_GridInfo.GridDimension.z / static_cast<float>(CUDA_META_GRID_GROUP_SIZE)));

		gridSize.x = m_GridInfo.GridDimension.x * cellSize;
		gridSize.y = m_GridInfo.GridDimension.y * cellSize;
		gridSize.z = m_GridInfo.GridDimension.z * cellSize;

		m_GridInfo.GridDelta.x = m_GridInfo.GridDimension.x / gridSize.x;
		m_GridInfo.GridDelta.y = m_GridInfo.GridDimension.y / gridSize.y;
		m_GridInfo.GridDelta.z = m_GridInfo.GridDimension.z / gridSize.z;

		const unsigned int numberOfCells = m_GridInfo.MetaGridDimension.x * m_GridInfo.MetaGridDimension.y * m_GridInfo.MetaGridDimension.z * CUDA_META_GRID_BLOCK_SIZE;

		d_TempSortIndices.resize(m_GridInfo.ParticleCount);
		d_ParticleCellIndices.resize(m_ParticleCount);
		d_SortIndices.resize(m_ParticleCount);
		d_ReversedSortIndices.resize(m_ParticleCount);
		d_CellOffsets.resize(numberOfCells);
		d_CellParticleCounts.resize(numberOfCells);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		cudaMemset(ComputeHelper::GetPointer(d_CellParticleCounts), 0, ComputeHelper::GetSizeInBytes(d_CellParticleCounts));

		COMPUTE_SAFE(cudaDeviceSynchronize())

		InsertParticlesMortonKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			m_GridInfo,
			m_Particles,
			ComputeHelper::GetPointer(d_ParticleCellIndices),
			ComputeHelper::GetPointer(d_CellParticleCounts),
			ComputeHelper::GetPointer(d_TempSortIndices)
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		thrust::exclusive_scan(
			d_CellParticleCounts.begin(),
			d_CellParticleCounts.end(),
			d_CellOffsets.begin()
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		CountingSortIndicesKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			m_GridInfo,
			ComputeHelper::GetPointer(d_ParticleCellIndices),
			ComputeHelper::GetPointer(d_CellOffsets),
			ComputeHelper::GetPointer(d_TempSortIndices),
			ComputeHelper::GetPointer(d_SortIndices)
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		auto& tempSequence = d_TempSortIndices;
		thrust::sequence(tempSequence.begin(), tempSequence.end());

		thrust::gather(
			d_SortIndices.begin(),
			d_SortIndices.end(),
			tempSequence.begin(),
			d_ReversedSortIndices.begin()
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	void ParticleSearch::ComputeNeighborhood()
	{
		d_NeighborCounts.resize(m_ParticleCount);

		ComputeCountsKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			m_Particles,
			m_GridInfo,
			ComputeHelper::GetPointer(d_CellOffsets),
			ComputeHelper::GetPointer(d_CellParticleCounts),
			ComputeHelper::GetPointer(d_NeighborCounts),
			ComputeHelper::GetPointer(d_ReversedSortIndices)
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		d_NeighborWriteOffsets.resize(m_ParticleCount);

		thrust::exclusive_scan(
			d_NeighborCounts.begin(),
			d_NeighborCounts.end(),
			d_NeighborWriteOffsets.begin()
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		unsigned int lastOffset = 0;
		ComputeHelper::MemcpyDeviceToHost(ComputeHelper::GetPointer(d_NeighborWriteOffsets) + m_ParticleCount - 1, &lastOffset, 1);
		unsigned int lastParticleNeighborCount = 0;
		ComputeHelper::MemcpyDeviceToHost(ComputeHelper::GetPointer(d_NeighborCounts) + m_ParticleCount - 1, &lastParticleNeighborCount, 1);
		const unsigned int totalNeighborCount = lastOffset + lastParticleNeighborCount;
		d_Neighbors.resize(totalNeighborCount);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		NeighborhoodQueryWithCountsKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			m_Particles,
			m_GridInfo,
			ComputeHelper::GetPointer(d_CellOffsets),
			ComputeHelper::GetPointer(d_CellParticleCounts),
			ComputeHelper::GetPointer(d_NeighborWriteOffsets),
			ComputeHelper::GetPointer(d_Neighbors),
			ComputeHelper::GetPointer(d_ReversedSortIndices)
		);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		auto* temp = new NeighborSet();
		temp->Neighbors = ComputeHelper::GetPointer(d_Neighbors);
		temp->Counts = ComputeHelper::GetPointer(d_NeighborCounts);
		temp->Offsets = ComputeHelper::GetPointer(d_NeighborWriteOffsets);

		COMPUTE_SAFE(cudaMemcpy(d_NeighborSet, temp, sizeof(NeighborSet), cudaMemcpyHostToDevice))

		COMPUTE_SAFE(cudaDeviceSynchronize())
	}
}