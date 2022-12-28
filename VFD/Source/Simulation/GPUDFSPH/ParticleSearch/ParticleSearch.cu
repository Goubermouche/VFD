#include "pch.h"
#include "ParticleSearch.h"

#include "ParticleSearchKernels.cuh"

#include <thrust/sequence.h>
#include <thrust/gather.h>

namespace vfd
{
	void ParticleSearch::Sort(DFSPHParticle* particles)
	{
		DFSPHParticle* temp;
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&temp), m_ParticleCount * sizeof(DFSPHParticle)))
		COMPUTE_SAFE(cudaMemcpy(temp, particles, m_ParticleCount * sizeof(DFSPHParticle), cudaMemcpyDeviceToDevice))

		PointSortKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, temp, ComputeHelper::GetPointer(d_SortIndices), m_ParticleCount);

		COMPUTE_SAFE(cudaDeviceSynchronize())
		COMPUTE_SAFE(cudaFree(temp))
	}

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
		const glm::ivec3 maxCell = data[1];

		m_Min.x = minCell.x * m_SearchRadius;
		m_Min.y = minCell.y * m_SearchRadius;
		m_Min.z = minCell.z * m_SearchRadius;

		m_Max.x = maxCell.x * m_SearchRadius;
		m_Max.y = maxCell.y * m_SearchRadius;
		m_Max.z = maxCell.z * m_SearchRadius;
	}

	void ParticleSearch::ComputeCellInformation()
	{
		SearchInfo gridInfo;
		gridInfo.ParticleCount = m_ParticleCount;
		gridInfo.SquaredSearchRadius = m_SearchRadius * m_SearchRadius;
		gridInfo.GridMin = m_Min;

		const float cellSize = m_SearchRadius;
		glm::vec3 gridSize = m_Max - m_Min;
		gridInfo.GridDimension.x = static_cast<unsigned int>(ceil(gridSize.x / cellSize));
		gridInfo.GridDimension.y = static_cast<unsigned int>(ceil(gridSize.y / cellSize));
		gridInfo.GridDimension.z = static_cast<unsigned int>(ceil(gridSize.z / cellSize));

		gridInfo.GridDimension.x += 4;
		gridInfo.GridDimension.y += 4;
		gridInfo.GridDimension.z += 4;
		gridInfo.GridMin -= glm::vec3(cellSize, cellSize, cellSize) * 2.0f;

		gridInfo.MetaGridDimension.x = static_cast<unsigned int>(ceil(gridInfo.GridDimension.x / static_cast<float>(CUDA_META_GRID_GROUP_SIZE)));
		gridInfo.MetaGridDimension.y = static_cast<unsigned int>(ceil(gridInfo.GridDimension.y / static_cast<float>(CUDA_META_GRID_GROUP_SIZE)));
		gridInfo.MetaGridDimension.z = static_cast<unsigned int>(ceil(gridInfo.GridDimension.z / static_cast<float>(CUDA_META_GRID_GROUP_SIZE)));

		gridSize.x = gridInfo.GridDimension.x * cellSize;
		gridSize.y = gridInfo.GridDimension.y * cellSize;
		gridSize.z = gridInfo.GridDimension.z * cellSize;

		gridInfo.GridDelta.x = gridInfo.GridDimension.x / gridSize.x;
		gridInfo.GridDelta.y = gridInfo.GridDimension.y / gridSize.y;
		gridInfo.GridDelta.z = gridInfo.GridDimension.z / gridSize.z;

		const unsigned int numberOfCells = gridInfo.MetaGridDimension.x * gridInfo.MetaGridDimension.y * gridInfo.MetaGridDimension.z * CUDA_META_GRID_BLOCK_SIZE;
		m_GridInfo = gridInfo;

		d_TempSortIndices.resize(gridInfo.ParticleCount);
		d_ParticleCellIndices.resize(m_ParticleCount);
		d_SortIndices.resize(m_ParticleCount);
		d_ReversedSortIndices.resize(m_ParticleCount);
		d_CellOffsets.resize(numberOfCells);
		d_CellParticleCounts.resize(numberOfCells);

		COMPUTE_SAFE(cudaDeviceSynchronize())

		cudaMemset(ComputeHelper::GetPointer(d_CellParticleCounts), 0, ComputeHelper::GetSizeInBytes(d_CellParticleCounts));

		COMPUTE_SAFE(cudaDeviceSynchronize())

		InsertParticlesMortonKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			gridInfo,
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
			gridInfo,
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
