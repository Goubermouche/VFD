#include "pch.h"
#include "PointSetImplementation.h"

namespace vfdcu {
	PointSetImplementation::PointSetImplementation(size_t particleCount, glm::vec3* particles)
	{
		m_ParticleCount = particleCount;
		m_Particles = particles;
		unsigned int threadStarts = 0;
		m_ThreadsPerBlock = 64;

		CudaHelper::GetThreadBlocks(static_cast<unsigned int>(particleCount), m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);
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
		CudaHelper::MemcpyHostToDevice(m_Particles, CudaHelper::GetPointer(d_Particles), m_ParticleCount);
	}
}