#ifndef GPU_DFSPH_SIMULATOR_H
#define GPU_DFSPH_SIMULATOR_H

#include "DFSPHImplementation.h"
#include "Renderer/Renderer.h"
#include "CollisionMap/RigidBodyObject.h"

namespace vfd
{
	struct GPUDFSPHSimulationDescription
	{
		
	};

	class DFSPHImplementation;

	// Wrapper around the GPU DFSPH simulator interface
	class GPUDFSPHSimulation : public RefCounted
	{
	public:
		GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc);

		void OnUpdate();
		const Ref<VertexArray>& GetVertexArray();

		unsigned int GetParticleCount()
		{
			return m_Implementation->GetParticleCount();
		}

		float GetParticleRadius()
		{
			return m_Implementation->GetParticleRadius();
		}

		std::vector<Ref<RigidBody>>& GetRigidBodies()
		{
			return m_RigidBodies;
		}

		Ref<Material>& GetRigidBodyMaterial()
		{
			return m_RigidBodyMaterial;
		}
	public:
		bool paused = false;
	private:
		GPUDFSPHSimulationDescription m_Description;
		Ref<DFSPHImplementation> m_Implementation;

		std::vector<Ref<RigidBody>> m_RigidBodies;
		Ref<Material> m_RigidBodyMaterial; // TEMP

		bool m_Initialized = false;
	};
}

#endif