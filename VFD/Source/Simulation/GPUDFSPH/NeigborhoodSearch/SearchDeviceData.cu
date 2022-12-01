#include "SearchDeviceData.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>

#include "PointSetImplementation.h"
#include "GridInfo.h"
#include "Utils/cuda_helper.h"
#include "cuNSearchKernels.cuh"

namespace vfdcu {
	void SearchDeviceData::ComputeMinMax(PointSet& pointSet) {
		if (pointSet.GetPointCount() == 0)
		{
			return;
		}

		const auto pointSetImpl = pointSet.m_Implementation.get();

		glm::ivec3 data[2];
		data[0] = glm::ivec3(std::numeric_limits<int>().max(), std::numeric_limits<int>().max(), std::numeric_limits<int>().max());
		data[1] = glm::ivec3(std::numeric_limits<int>().min(), std::numeric_limits<int>().min(), std::numeric_limits<int>().min());
		d_MinMax.resize(2);
		CudaHelper::MemcpyHostToDevice(data, CudaHelper::GetPointer(d_MinMax), 2);

		ComputeMinMaxKernel << <pointSetImpl->m_BlockStartsForParticles, pointSetImpl->m_ThreadsPerBlock >> > (
			(glm::vec3*)CudaHelper::GetPointer(pointSetImpl->d_Particles),
			static_cast<unsigned int>(pointSet.GetPointCount()),
			m_SearchRadius,
			CudaHelper::GetPointer(d_MinMax),
			CudaHelper::GetPointer(d_MinMax) + 1
		);

		COMPUTE_CHECK("kernel failed");
		COMPUTE_SAFE(cudaDeviceSynchronize());

		CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(d_MinMax), data, 2);
		glm::ivec3 minCell = data[0];
		glm::ivec3 maxCell = data[1];

		pointSetImpl->m_Min.x = minCell.x * m_SearchRadius;
		pointSetImpl->m_Min.y = minCell.y * m_SearchRadius;
		pointSetImpl->m_Min.z = minCell.z * m_SearchRadius;

		pointSetImpl->m_Max.x = maxCell.x * m_SearchRadius;
		pointSetImpl->m_Max.y = maxCell.y * m_SearchRadius;
		pointSetImpl->m_Max.z = maxCell.z * m_SearchRadius;
	}

	void SearchDeviceData::ComputeCellInformation(PointSet& pointSet) {
		if (pointSet.GetPointCount() == 0)
		{
			return;
		}

		auto pointSetImpl = pointSet.m_Implementation.get();
		glm::vec3 sceneMin = pointSetImpl->m_Min;
		glm::vec3 sceneMax = pointSetImpl->m_Max;

		GridInfo gridInfo;
		gridInfo.ParticleCount = static_cast<unsigned int>(pointSet.GetPointCount());
		gridInfo.SquaredSearchRadius = m_SearchRadius * m_SearchRadius;
		gridInfo.GridMin = sceneMin;

		float cellSize = m_SearchRadius;
		glm::vec3 gridSize = sceneMax - sceneMin;
		gridInfo.GridDimension.x = static_cast<unsigned int>(ceil(gridSize.x / cellSize));
		gridInfo.GridDimension.y = static_cast<unsigned int>(ceil(gridSize.y / cellSize));
		gridInfo.GridDimension.z = static_cast<unsigned int>(ceil(gridSize.z / cellSize));

		//Increase grid by 2 cells in each direciton (+4 in each dimension) to skip bounds checks in the kernel
		gridInfo.GridDimension.x += 4;
		gridInfo.GridDimension.y += 4;
		gridInfo.GridDimension.z += 4;
		gridInfo.GridMin -= glm::vec3(cellSize, cellSize, cellSize) * (float)2;

		//One meta grid cell contains 8x8x8 grild cells. (512)
		gridInfo.MetaGridDimension.x = static_cast<unsigned int>(ceil(gridInfo.GridDimension.x / (float)CUDA_META_GRID_GROUP_SIZE));
		gridInfo.MetaGridDimension.y = static_cast<unsigned int>(ceil(gridInfo.GridDimension.y / (float)CUDA_META_GRID_GROUP_SIZE));
		gridInfo.MetaGridDimension.z = static_cast<unsigned int>(ceil(gridInfo.GridDimension.z / (float)CUDA_META_GRID_GROUP_SIZE));

		// Adjust grid size to multiple of cell size
		gridSize.x = gridInfo.GridDimension.x * cellSize;
		gridSize.y = gridInfo.GridDimension.y * cellSize;
		gridSize.z = gridInfo.GridDimension.z * cellSize;

		gridInfo.GridDelta.x = gridInfo.GridDimension.x / gridSize.x;
		gridInfo.GridDelta.y = gridInfo.GridDimension.y / gridSize.y;
		gridInfo.GridDelta.z = gridInfo.GridDimension.z / gridSize.z;

		d_TempSortIndices.resize(gridInfo.ParticleCount);

		unsigned int numberOfCells = (gridInfo.MetaGridDimension.x * gridInfo.MetaGridDimension.y * gridInfo.MetaGridDimension.z) * CUDA_META_GRID_BLOCK_SIZE;
		pointSet.m_Implementation->PrepareInternalDataStructures(gridInfo, numberOfCells);

		COMPUTE_CHECK("error");
		COMPUTE_SAFE(cudaDeviceSynchronize());

		cudaMemset(CudaHelper::GetPointer(pointSetImpl->d_CellParticleCounts), 0, CudaHelper::GetSizeInBytes(pointSetImpl->d_CellParticleCounts));

		COMPUTE_CHECK("error");
		COMPUTE_SAFE(cudaDeviceSynchronize());

		InsertParticlesMortonKernel << <pointSetImpl->m_BlockStartsForParticles, pointSetImpl->m_ThreadsPerBlock >> > (
			gridInfo,
			(glm::vec3*)CudaHelper::GetPointer(pointSetImpl->d_Particles),
			CudaHelper::GetPointer(pointSetImpl->d_ParticleCellIndices),
			CudaHelper::GetPointer(pointSetImpl->d_CellParticleCounts),
			CudaHelper::GetPointer(d_TempSortIndices)
			);

		COMPUTE_CHECK("error");
		COMPUTE_SAFE(cudaDeviceSynchronize());

		thrust::exclusive_scan(
			pointSetImpl->d_CellParticleCounts.begin(),
			pointSetImpl->d_CellParticleCounts.end(),
			pointSetImpl->d_CellOffsets.begin());

		COMPUTE_SAFE(cudaDeviceSynchronize());

		CountingSortIndicesKernel << <pointSetImpl->m_BlockStartsForParticles, pointSetImpl->m_ThreadsPerBlock >> > (
			gridInfo,
			CudaHelper::GetPointer(pointSetImpl->d_ParticleCellIndices),
			CudaHelper::GetPointer(pointSetImpl->d_CellOffsets),
			CudaHelper::GetPointer(d_TempSortIndices),
			CudaHelper::GetPointer(pointSetImpl->d_SortIndices)
		);

		COMPUTE_SAFE(cudaDeviceSynchronize());

		auto& tempSequence = d_TempSortIndices;
		thrust::sequence(tempSequence.begin(), tempSequence.end());

		thrust::gather(
			pointSetImpl->d_SortIndices.begin(),
			pointSetImpl->d_SortIndices.end(),
			tempSequence.begin(),
			pointSetImpl->d_ReversedSortIndices.begin());


		COMPUTE_CHECK("error");
		COMPUTE_SAFE(cudaDeviceSynchronize());

		pointSet.m_SortedIndices.resize(pointSetImpl->d_SortIndices.size());
		CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(pointSetImpl->d_SortIndices), pointSet.m_SortedIndices.data(), pointSetImpl->d_SortIndices.size());
	}

	void SearchDeviceData::ComputeNeighborhood(PointSet& queryPointSet, PointSet& pointSet, unsigned int neighborListEntry) {
		if (queryPointSet.GetPointCount() == 0)
		{
			return;
		}

		auto queryPointSetImpl = queryPointSet.m_Implementation.get();
		auto pointSetImpl = pointSet.m_Implementation.get();

		unsigned int particleCount = static_cast<unsigned int>(queryPointSet.GetPointCount());

		d_NeighborCounts.resize(particleCount);

		ComputeCountsKernel << <queryPointSetImpl->m_BlockStartsForParticles, queryPointSetImpl->m_ThreadsPerBlock >> > (
			(glm::vec3*)CudaHelper::GetPointer(queryPointSetImpl->d_Particles),
			static_cast<unsigned int>(queryPointSet.GetPointCount()),

			pointSetImpl->m_GridInfo,
			(glm::vec3*)CudaHelper::GetPointer(pointSetImpl->d_Particles),
			CudaHelper::GetPointer(pointSetImpl->d_CellOffsets),
			CudaHelper::GetPointer(pointSetImpl->d_CellParticleCounts),

			CudaHelper::GetPointer(d_NeighborCounts),
			CudaHelper::GetPointer(pointSetImpl->d_ReversedSortIndices)
		);

		COMPUTE_CHECK("error");
		COMPUTE_SAFE(cudaDeviceSynchronize());

		d_NeighborWriteOffsets.resize(particleCount);

		//Prefix sum over neighbor counts
		thrust::exclusive_scan(
			d_NeighborCounts.begin(),
			d_NeighborCounts.end(),
			d_NeighborWriteOffsets.begin());

		COMPUTE_SAFE(cudaDeviceSynchronize());

		unsigned int lastOffset = 0;
		CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(d_NeighborWriteOffsets) + particleCount - 1, &lastOffset, 1);
		unsigned int lastParticleNeighborCount = 0;
		CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(d_NeighborCounts) + particleCount - 1, &lastParticleNeighborCount, 1);
		unsigned int totalNeighborCount = lastOffset + lastParticleNeighborCount;
		d_Neighbors.resize(totalNeighborCount);

		COMPUTE_SAFE(cudaDeviceSynchronize());

		NeighborhoodQueryWithCountsKernel << <queryPointSetImpl->m_BlockStartsForParticles, queryPointSetImpl->m_ThreadsPerBlock >> > (
			(glm::vec3*)CudaHelper::GetPointer(queryPointSetImpl->d_Particles),
			static_cast<unsigned int>(queryPointSet.GetPointCount()),

			pointSetImpl->m_GridInfo,
			(glm::vec3*)CudaHelper::GetPointer(pointSetImpl->d_Particles),
			CudaHelper::GetPointer(pointSetImpl->d_CellOffsets),
			CudaHelper::GetPointer(pointSetImpl->d_CellParticleCounts),

			CudaHelper::GetPointer(d_NeighborWriteOffsets),
			CudaHelper::GetPointer(d_Neighbors),
			CudaHelper::GetPointer(pointSetImpl->d_ReversedSortIndices)
		);

		COMPUTE_CHECK("error");
		COMPUTE_SAFE(cudaDeviceSynchronize());
		
		auto& neighborSet = queryPointSet.m_Neighbors[neighborListEntry];

		if (neighborSet.NeighborCountAllocationSize < totalNeighborCount)
		{
			if (neighborSet.NeighborCountAllocationSize != 0)
			{
				cudaFreeHost(neighborSet.Neighbors);
			}

			neighborSet.NeighborCountAllocationSize = static_cast<unsigned int>(totalNeighborCount * 1.5);
			cudaMallocHost(&neighborSet.Neighbors, sizeof(unsigned int) * neighborSet.NeighborCountAllocationSize);
		}
		if (neighborSet.ParticleCountAllocationSize < particleCount)
		{
			if (neighborSet.ParticleCountAllocationSize != 0)
			{
				cudaFreeHost(neighborSet.Offsets);
				cudaFreeHost(neighborSet.Counts);
			}

			neighborSet.ParticleCountAllocationSize = static_cast<unsigned int>(particleCount * 1.5);
			cudaMallocHost(&neighborSet.Offsets, sizeof(unsigned int) * neighborSet.ParticleCountAllocationSize);
			cudaMallocHost(&neighborSet.Counts, sizeof(unsigned int) * neighborSet.ParticleCountAllocationSize);
		}

		CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(d_Neighbors), neighborSet.Neighbors, totalNeighborCount);
		CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(d_NeighborCounts), neighborSet.Counts, particleCount);
		CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(d_NeighborWriteOffsets), neighborSet.Offsets, particleCount);
	}
}