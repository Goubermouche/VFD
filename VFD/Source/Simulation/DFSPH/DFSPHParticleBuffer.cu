#include "pch.h"
#include "DFSPHParticleBuffer.h"

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
	}

	DFSPHParticleBuffer::~DFSPHParticleBuffer()
	{
		for (unsigned int i = 0u; i < m_FrameCount; i++)
		{
			delete m_Frames[i];
		}
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