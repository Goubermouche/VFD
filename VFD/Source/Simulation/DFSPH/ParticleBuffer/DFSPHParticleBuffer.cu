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
		m_VertexArray->AddVertexBuffer(m_VertexBuffer);
		m_VertexBuffer->Unbind();
		m_VertexArray->Unbind();

		m_Frames = std::vector<DFSPHParticleFrame>(m_FrameCount);
		COMPUTE_SAFE(cudaMalloc(&m_Buffer, sizeof(DFSPHParticleSimple) * m_ParticleCount));

		// Calculate block and thread counts
		unsigned int threadStarts = 0u;
		ComputeHelper::GetThreadBlocks(m_ParticleCount, m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);
	}

	DFSPHParticleBuffer::~DFSPHParticleBuffer()
	{
		COMPUTE_SAFE(cudaFree(m_Buffer));
	}

	void DFSPHParticleBuffer::SetFrameData(unsigned int frame, DFSPHParticle* particles)
	{
		DFSPHParticleFrame frameData;
		frameData.ParticleData = std::vector<DFSPHParticleSimple>(m_ParticleCount);

		ConvertParticlesToBuffer <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			m_Buffer,
			m_ParticleCount
		);

		COMPUTE_SAFE(cudaMemcpy(frameData.ParticleData.data(), m_Buffer, sizeof(DFSPHParticleSimple) * m_ParticleCount, cudaMemcpyDeviceToHost));
		m_Frames[frame] = frameData;
	}

	void DFSPHParticleBuffer::SetActiveFrame(unsigned int frame)
	{
		if(frame < m_FrameCount)
		{
			if(m_Frames[frame].ParticleData.empty() == false)
			{
				m_VertexBuffer->SetData(0, m_ParticleCount * sizeof(DFSPHParticleSimple), m_Frames[frame].ParticleData.data());

				//DFSPHParticleSimple* temp = new DFSPHParticleSimple[m_ParticleCount];
				//glGetBufferSubData(GL_ARRAY_BUFFER, 0, m_ParticleCount * sizeof(DFSPHParticleSimple), temp);

				//WARN(temp[0].Position.y)
				//WARN(m_Frames[frame].ParticleData[0].Position.y)
			}
			else
			{
				ERR("pointer null");
			}
		}
		else
		{
			ERR("frame out of range!");
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