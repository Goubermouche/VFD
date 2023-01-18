#ifndef DFSPH_PARTICLE_BUFFER_H
#define DFSPH_PARTICLE_BUFFER_H

#include "Renderer/Material.h"
#include "Renderer/VertexArray.h"
#include "Simulation/DFSPH/DFSPHKernels.cuh"

#include "Simulation/DFSPH/Structures/DFSPHParticle.h"
#include "Simulation/DFSPH/Structures/DFSPHParticleSimple.h"

namespace vfd
{
	struct DFSPHParticleFrame
	{
		std::vector<DFSPHParticleSimple> ParticleData;
	};

	class DFSPHParticleBuffer : public RefCounted
	{
	public:
		DFSPHParticleBuffer(unsigned int frameCount, unsigned int particleCount);
		~DFSPHParticleBuffer();

		void SetFrameData(unsigned int frame, DFSPHParticle* particles);
		void SetActiveFrame(unsigned int frame);

		const Ref<VertexArray>& GetVertexArray() const;
		const Ref<Material>& GetMaterial() const;
	private:
		friend class DFSPHImplementation;

		std::vector<DFSPHParticleFrame> m_Frames;
		DFSPHParticleSimple* m_Buffer;

		Ref<Material> m_Material;
		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;

		unsigned int m_FrameCount;
		unsigned int m_ParticleCount;

		int m_ThreadsPerBlock = MAX_CUDA_THREADS_PER_BLOCK;
		unsigned int m_BlockStartsForParticles;
	};
}

#endif // !DFSPH_PARTICLE_BUFFER_H