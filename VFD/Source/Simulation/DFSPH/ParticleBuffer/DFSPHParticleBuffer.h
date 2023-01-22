#ifndef DFSPH_PARTICLE_BUFFER_H
#define DFSPH_PARTICLE_BUFFER_H

#include "Renderer/Material.h"
#include "Renderer/VertexArray.h"
#include "Simulation/DFSPH/DFSPHKernels.cuh"

#include "Simulation/DFSPH/Structures/DFSPHParticle.h"
#include "Simulation/DFSPH/Structures/DFSPHParticleSimple.h"

namespace vfd
{
	struct DFSPHParticleFrame : public RefCounted
	{
		std::vector<DFSPHParticleSimple> ParticleData;

		float MaxVelocityMagnitude;
		float CurrentTimeStep;
	};

	class DFSPHParticleBuffer : public RefCounted
	{
	public:
		DFSPHParticleBuffer(unsigned int frameCount, unsigned int particleCount);

		void SetFrameData(unsigned int frameIndex, const Ref<DFSPHParticleFrame>& frame);
		void SetActiveFrame(unsigned int frame);

		const Ref<DFSPHParticleFrame>& GetActiveFrame() const;
		const Ref<DFSPHParticleFrame>& GetFrame(unsigned int frameIndex) const;
		const Ref<VertexArray>& GetVertexArray() const;
		unsigned int GetActiveFrameIndex() const;
	private:
		friend class DFSPHImplementation;

		std::vector<Ref<DFSPHParticleFrame>> m_Frames;

		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;

		unsigned int m_FrameCount;
		unsigned int m_ParticleCount;
		int m_FrameIndex = 0u;
	};
}

#endif // !DFSPH_PARTICLE_BUFFER_H