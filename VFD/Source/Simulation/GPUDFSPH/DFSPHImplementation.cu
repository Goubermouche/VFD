#include "pch.h"
#include "DFSPHImplementation.h"

#include <cuda_gl_interop.h>

#include "Compute/ComputeHelper.h"
#include "DFSPHKernels.cuh"

namespace vfd
{
#define N 3
	DFSPHImplementation::DFSPHImplementation()
	{
		m_Particles = new DFSPHParticle[N];

		for (size_t i = 0; i < N; i++)
		{
			m_Particles[i] = DFSPHParticle{ glm::vec3(i), glm::vec3(1, 0, 0) };
		}

		m_VertexArray = Ref<VertexArray>::Create();
		m_VertexBuffer = Ref<VertexBuffer>::Create(N * sizeof(DFSPHParticle));

		m_VertexBuffer->SetLayout({
			{ShaderDataType::Float3, "a_Position"},
			{ShaderDataType::Float3, "a_Velocity"}
		});

		m_VertexArray->AddVertexBuffer(m_VertexBuffer);

		m_VertexBuffer->SetData(0, N * sizeof(DFSPHParticle), m_Particles);
		m_VertexBuffer->Unbind();
		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_VertexBuffer->GetRendererID()));
	}

	DFSPHImplementation::~DFSPHImplementation()
	{
		delete[] m_Particles;
		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_VertexBuffer->GetRendererID()));
	}

	void DFSPHImplementation::OnUpdate()
	{
		m_DeviceDataUpdated = true;

		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject((void**)&particles, m_VertexBuffer->GetRendererID()));

		TestKernel << < 1, 3 >> > (particles);
		COMPUTE_SAFE(cudaDeviceSynchronize());

		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()));
	}

	const Ref<VertexArray>& DFSPHImplementation::GetVertexArray() const
	{
		return m_VertexArray;
	}
}