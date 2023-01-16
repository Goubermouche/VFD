#ifndef DFSPH_PARTICLE_BUFFER_H
#define DFSPH_PARTICLE_BUFFER_H

#include "Renderer/Material.h"
#include "Renderer/VertexArray.h"

#include "Simulation/DFSPH/Structures/DFSPHParticle.h"
#include "Simulation/DFSPH/Structures/DFSPHParticleSimple.h"

namespace vfd
{
	class DFSPHParticleBuffer : public RefCounted
	{
	public:
		DFSPHParticleBuffer(unsigned int frameCount, unsigned int particleCount);
		~DFSPHParticleBuffer();

		void SetActiveFrame(unsigned int frame);

		const Ref<VertexArray>& GetVertexArray() const;
		const Ref<Material>& GetMaterial() const;
	private:
		friend class DFSPHImplementation;

		std::vector<DFSPHParticleSimple*> m_Frames;

		Ref<Material> m_Material;
		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;

		unsigned int m_FrameCount;
		unsigned int m_ParticleCount;
	};
}

#endif // !DFSPH_PARTICLE_BUFFER_H