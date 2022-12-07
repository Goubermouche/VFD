#ifndef DFSPH_IMPLEMENTATION_H
#define DFSPH_IMPLEMENTATION_H

#include "pch.h"
#include "Renderer/VertexArray.h"
#include "DFSPHParticle.h"
#include "DFSPHSimulationInfo.h"
#include "NeigborhoodSearch/NeighborhoodSearchP.h"

namespace vfd
{
	class DFSPHImplementation : public RefCounted
	{
	public:
		DFSPHImplementation();
		~DFSPHImplementation();

		const Ref<VertexArray>& GetVertexArray() const;

		void OnUpdate();
	private:
		DFSPHParticle* m_Particles;

		DFSPHSimulationInfo m_Info; // main? 
		DFSPHSimulationInfo* d_Info;

		NeighborhoodSearch* m_NeighborhoodSearch;

		Ref<VertexArray> m_VertexArray;
		Ref<VertexBuffer> m_VertexBuffer;

		unsigned int m_IterationCount = 0;
	};
}

#endif // !DFSPH_IMPLEMENTATION_H