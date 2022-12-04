#ifndef DFSPH_IMPLEMENTATION_H
#define DFSPH_IMPLEMENTATION_H

#include "pch.h"
#include "Renderer/VertexArray.h"
#include "DFSPHParticle.h"

namespace vfd
{
	class DFSPHImplementation
	{
	public:
		DFSPHImplementation();
		~DFSPHImplementation();

		const Ref<VertexArray>& GetVertexArray() const;

		void OnUpdate();
	private:
		DFSPHParticle* m_Particles;

		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;

		bool m_DeviceDataUpdated = false;
	};
}

#endif // !DFSPH_IMPLEMENTATION_H