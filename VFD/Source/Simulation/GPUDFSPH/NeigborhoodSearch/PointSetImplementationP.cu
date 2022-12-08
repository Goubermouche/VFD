#include "pch.h"
#include "PointSetImplementationP.h"

#include "NeighborhoodSearchKernelsP.cuh"

namespace vfd {
	PointSetImplementation::PointSetImplementation(size_t particleCount, DFSPHParticle* particles)
	{
		m_ParticleCount = particleCount;
		m_Particles = particles;
		unsigned int threadStarts = 0;
		m_ThreadsPerBlock = 64;

		vfd::ComputeHelper::GetThreadBlocks(static_cast<unsigned int>(particleCount), m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);
		CopyToDevice();
	}

	PointSetImplementation& PointSetImplementation::operator=(PointSetImplementation const& other)
	{
		if (this != &other)
		{
			PointSetImplementation tmp(other);
			std::swap(tmp, *this);
		}

		return *this;
	}

	void PointSetImplementation::PrepareInternalDataStructures(GridInfo& gridInfo, size_t numberOfCells)
	{
		this->m_GridInfo = gridInfo;

		d_ParticleCellIndices.resize(m_ParticleCount);
		d_SortIndices.resize(m_ParticleCount);
		d_ReversedSortIndices.resize(m_ParticleCount);
		d_CellOffsets.resize(numberOfCells);
		d_CellParticleCounts.resize(numberOfCells);
	}

	void PointSetImplementation::CopyToDevice()
	{
		d_Particles.resize(m_ParticleCount);
		vfd::ComputeHelper::MemcpyHostToDevice(m_Particles, vfd::ComputeHelper::GetPointer(d_Particles), m_ParticleCount);
	}

	void PointSetImplementation::SortField(DFSPHParticle* particles)
	{
		// Naive implementation
		// TODO: have a local buffer that stores the temp data so we don't need to allocate it every time we sort
		DFSPHParticle* d_TempPoints;
		COMPUTE_SAFE(cudaMalloc((void**)&d_TempPoints, 3 * sizeof(DFSPHParticle)))
		COMPUTE_SAFE(cudaMemcpy(d_TempPoints, particles, 3 * sizeof(DFSPHParticle), cudaMemcpyDeviceToDevice))

		// vfd::ComputeHelper::GetThreadBlocks(static_cast<unsigned int>(particleCount), m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);

		SortKernelP << <1, 3 >> > (particles, d_TempPoints, ComputeHelper::GetPointer(d_SortIndices), m_ParticleCount);
		COMPUTE_SAFE(cudaThreadSynchronize())

		COMPUTE_SAFE(cudaFree(d_TempPoints))
	}
}
