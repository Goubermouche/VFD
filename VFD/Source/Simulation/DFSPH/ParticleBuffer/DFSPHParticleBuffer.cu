#include "pch.h"
#include "DFSPHParticleBuffer.h"

#include "Compute/ComputeHelper.h"

namespace vfd
{
	DFSPHParticleBuffer::DFSPHParticleBuffer(unsigned int frameCount, unsigned int particleCount)
		: m_FrameCount(frameCount), m_ParticleCount(particleCount)
	{
		m_VertexArray = Ref<VertexArray>::Create();
		m_VertexBuffer = Ref<VertexBuffer>::Create(particleCount * sizeof(DFSPHParticleSimple));
		m_VertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position"     },
			{ ShaderDataType::Float3, "a_Velocity"     },
			{ ShaderDataType::Float3, "a_Acceleration" }
		});
		m_VertexArray->AddVertexBuffer(m_VertexBuffer);
		m_VertexBuffer->Unbind();
		m_VertexArray->Unbind();

		m_Frames = std::vector<Ref<DFSPHParticleFrame>>(m_FrameCount);
		m_FrameIndex = -1;
	}

	void DFSPHParticleBuffer::SetFrameData(unsigned int frameIndex, const Ref<DFSPHParticleFrame>& frame)
	{
		m_Frames[frameIndex] = frame;
	}

	void DFSPHParticleBuffer::SetActiveFrame(unsigned int frame)
	{
		if(frame < m_FrameCount && static_cast<int>(frame) != m_FrameIndex)
		{
			if(m_Frames[frame])
			{
				m_VertexBuffer->SetData(0u, m_ParticleCount * sizeof(DFSPHParticleSimple), m_Frames[frame]->ParticleData.data());
				m_FrameIndex = static_cast<int>(frame);
			}
		}
	}

	const Ref<DFSPHParticleFrame>& DFSPHParticleBuffer::GetActiveFrame() const
	{
		return m_Frames[m_FrameIndex];
	}

	const Ref<DFSPHParticleFrame>& DFSPHParticleBuffer::GetFrame(unsigned int frameIndex) const
	{
		return m_Frames[frameIndex];
	}

	const Ref<VertexArray>& DFSPHParticleBuffer::GetVertexArray() const
	{
		return m_VertexArray;
	}

	unsigned int DFSPHParticleBuffer::GetActiveFrameIndex() const
	{
		return m_FrameIndex;
	}
}