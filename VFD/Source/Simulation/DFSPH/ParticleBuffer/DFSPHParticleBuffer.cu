#include "pch.h"
#include "DFSPHParticleBuffer.h"

#include "Compute/ComputeHelper.h"
#include "Simulation/DFSPH/ParticleBuffer/DFSPHParticleBufferKernels.cuh"

namespace vfd
{
	DFSPHParticleBuffer::DFSPHParticleBuffer(unsigned int frameCount, unsigned int particleCount)
		: m_FrameCount(frameCount), m_ParticleCount(particleCount)
	{
		m_VertexArray = Ref<VertexArray>::Create();
		m_VertexBuffer = Ref<VertexBuffer>::Create(particleCount * sizeof(DFSPHParticleSimple));
		m_VertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position" }
		});

		m_Frames = std::vector<DFSPHParticleSimple*>(m_FrameCount);
		COMPUTE_SAFE(cudaMalloc((void**)&m_Buffer, sizeof(DFSPHParticleSimple) * m_ParticleCount))

		// Calculate block and thread counts
		unsigned int threadStarts = 0u;
		ComputeHelper::GetThreadBlocks(m_ParticleCount, m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);
	}

	DFSPHParticleBuffer::~DFSPHParticleBuffer()
	{
		for (unsigned int i = 0u; i < m_FrameCount; i++)
		{
			delete m_Frames[i];
		}

		COMPUTE_SAFE(cudaFree(m_Buffer))
	}

	void DFSPHParticleBuffer::SetFrameData(unsigned int frame, DFSPHParticle* particles)
	{
		m_Frames[frame] = new DFSPHParticleSimple[m_ParticleCount];

		ConvertParticlesToBuffer <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			m_Buffer,
			m_ParticleCount
		);

		COMPUTE_SAFE(cudaMemcpy(m_Frames[frame], m_Buffer, sizeof(DFSPHParticleSimple) * m_ParticleCount, cudaMemcpyDeviceToHost))
	}

	void DFSPHParticleBuffer::SetActiveFrame(unsigned int frame)
	{
		if(frame >= 0u && frame < m_FrameCount)
		{
			m_VertexBuffer->SetData(m_Frames[frame]);
		}
		else
		{
			ERR("frame out of range!")
		}
	}

	const Ref<VertexArray>& DFSPHParticleBuffer::GetVertexArray() const
	{
		return m_VertexArray;
	}

	const Ref<Material>& DFSPHParticleBuffer::GetMaterial() const
	{
		return m_Material;
	}
}